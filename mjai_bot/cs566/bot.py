import json
import sys
import time
from typing import Optional, Dict, List, Tuple

import torch
import torch.nn.functional as F
import pathlib
from .logger import logger

try:
    from .model import MahjongResNet
    from .gamestate import RoundState, pai_to_idx, idx_to_pai
except ImportError:
    from model import MahjongResNet
    from gamestate import RoundState, pai_to_idx, idx_to_pai


CALL_CLASS_NAMES = ["pass", "chi", "pon", "hora", "dmk", "ank", "kak", "rii"]

CALL_THRESHOLDS = {
    1: 0.80,  # chi
    2: 0.70,  # pon
    4: 0.99,  # daiminkan
    5: 0.96,  # ankan
    6: 0.94,  # kakan
}

# Akagi / UI action ordering for 4p
ACTION_ORDER_4P = [
    "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
    "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
    "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
    "E", "S", "W", "N", "P", "F", "C",
    "5mr", "5pr", "5sr",
    "reach", "chi_low", "chi_mid", "chi_high", "pon", "kan_select", "hora", "ryukyoku", "none"
]
ACTION_TO_IDX_4P = {a: i for i, a in enumerate(ACTION_ORDER_4P)}

RED_ACTION_BY_TILE = {
    "5m": "5mr",
    "5p": "5pr",
    "5s": "5sr",
}


class Bot:
    def __init__(self, device: Optional[str] = None):
        self.player_id: Optional[int] = None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.round_state: Optional[RoundState] = None
        self.model_path = {
            "discard": "./best_discard.pt",
            "call": "./best_call.pt"
        }

    # ----------------------------------------------------------
    # Model loading
    # ----------------------------------------------------------
    def _load_model(self):
        if self.model is not None:
            return

        discard_path = pathlib.Path(self.model_path["discard"])
        call_path = pathlib.Path(self.model_path["call"])
        if not discard_path.exists() or not call_path.exists():
            discard_path = pathlib.Path(__file__).parent / self.model_path["discard"]
            call_path = pathlib.Path(__file__).parent / self.model_path["call"]
        if not discard_path.exists() or not call_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        model = MahjongResNet().to(self.device)
        model.discard_model.load_state_dict(torch.load(discard_path, map_location=self.device))
        model.call_model.load_state_dict(torch.load(call_path, map_location=self.device))
        model.eval()
        self.model = model

    def _unload_model(self):
        self.model = None
        if self.device == "cuda":
            torch.cuda.empty_cache()

    # ----------------------------------------------------------
    # Tensor helpers
    # ----------------------------------------------------------
    def _get_state_tensors(self, called_tile=None):
        x = self.round_state.to_feature(self.player_id)
        if called_tile:
            called_plane = torch.zeros(1, 34, dtype=torch.float32)
            called_plane[0, pai_to_idx(called_tile)] = 1.0
            x = torch.cat([x, called_plane], dim=0)
        x = x.unsqueeze(0).to(self.device)

        hist, hist_mask = self.round_state.get_history(self.player_id)
        hist = hist.unsqueeze(0).to(self.device)
        hist_mask = hist_mask.unsqueeze(0).to(self.device)
        return x, hist, hist_mask

    @staticmethod
    def _masked_prediction(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return logits.masked_fill(~mask, -1e9)

    def _masked_call_prediction(self, logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
        masked_logits = logits.clone()
        masked_logits[~legal_mask] = -1e9

        probs = F.softmax(masked_logits, dim=-1)

        for action_idx, threshold in CALL_THRESHOLDS.items():
            p = probs[:, action_idx]
            reject = p < threshold
            if reject.any():
                probs[reject, action_idx] = 0.0

        return probs

    # ----------------------------------------------------------
    # Forward helpers
    # ----------------------------------------------------------
    @torch.no_grad()
    def _forward_discard(self, x, hist, hist_mask):
        if hasattr(self.model, "forward_discard"):
            out = self.model.forward_discard(x, hist, hist_mask)
        elif hasattr(self.model, "discard_model"):
            out = self.model.discard_model(x, hist, hist_mask)
        else:
            raise AttributeError("Model has neither forward_discard nor discard_model.")
        return out[0] if isinstance(out, tuple) else out

    @torch.no_grad()
    def _forward_call(self, x, hist, hist_mask):
        if hasattr(self.model, "forward_call"):
            out = self.model.forward_call(x, hist, hist_mask)
        elif hasattr(self.model, "call_model"):
            out = self.model.call_model(x, hist, hist_mask)
        else:
            raise AttributeError("Model has neither forward_call nor call_model.")
        return out[0] if isinstance(out, tuple) else out

    # ----------------------------------------------------------
    # Meta helpers
    # ----------------------------------------------------------
    @staticmethod
    def _build_meta_from_legal_scores(
        legal_scores: Dict[str, float],
        eval_time_ns: int,
        is_greedy: bool = True,
    ) -> dict:
        """
        Build Akagi-style meta:
        - mask_bits: bit i means ACTION_ORDER_4P[i] is legal
        - q_values: packed in increasing global action index order
        """
        mask_bits = 0
        q_values: List[float] = []

        for action_idx, action_name in enumerate(ACTION_ORDER_4P):
            if action_name in legal_scores:
                mask_bits |= (1 << action_idx)
                q_values.append(float(legal_scores[action_name]))

        return {
            "q_values": q_values,
            "mask_bits": int(mask_bits),
            "is_greedy": bool(is_greedy),
            "eval_time_ns": int(eval_time_ns),
        }

    def _tile_to_ui_discard_action(self, pai: str) -> str:
        return RED_ACTION_BY_TILE.get(pai, pai) if pai in {"5m", "5p", "5s"} else pai

    def _hand_discard_actions_with_scores(self, logits: torch.Tensor, hand_mask: torch.Tensor) -> Dict[str, float]:
        """
        Convert discard logits [34] into legal UI actions.
        If both normal and red 5 are discardable, both get the same tile34 logit.
        """
        logits = logits.detach().cpu()
        legal_scores: Dict[str, float] = {}

        # Normal tile34 legality
        for tile_idx in range(34):
            if bool(hand_mask[tile_idx].item()):
                action_name = idx_to_pai(tile_idx)
                legal_scores[action_name] = float(logits[tile_idx].item())

        # Replace / augment with red-five discard actions based on actual hand contents
        hand_pais = list(self.round_state.hands[self.player_id])
        for red_pai in ("5mr", "5pr", "5sr"):
            if red_pai in hand_pais:
                base_pai = red_pai[:2]  # 5m / 5p / 5s
                tile_idx = pai_to_idx(base_pai)
                # Add red discard action with same tile34 logit
                legal_scores[red_pai] = float(logits[tile_idx].item())

        return legal_scores

    def _call_action_name_from_decision_and_event(self, decision: int, trigger_event: dict) -> Optional[str]:
        """
        Map internal call class to Akagi/UI action name.
        """
        etype = trigger_event["type"]

        if decision == 0:
            return "none"

        if decision == 3:
            return "hora"

        if decision == 2:
            return "pon"

        if decision in (4, 5, 6):
            return "kan_select"

        if decision == 7:
            return "reach"

        if decision == 1:
            if etype == "dahai" and trigger_event["actor"] != self.player_id:
                tile_idx = pai_to_idx(trigger_event["pai"])
                chi_kind = self._find_chi_kind(tile_idx)
                if chi_kind is not None:
                    return chi_kind
            return None

        return None

    def _call_legal_scores(self, logits, trigger_event, legal_mask):
        logits = logits.detach().cpu().view(-1)
        legal_mask = legal_mask.detach().cpu().view(-1)
        legal_scores = {}

        for cls_idx in range(len(CALL_CLASS_NAMES)):
            if not bool(legal_mask[cls_idx].item()):
                continue

            score = float(logits[cls_idx].item())

            if cls_idx == 1:  # chi
                if trigger_event["type"] == "dahai" and trigger_event["actor"] != self.player_id:
                    tile_idx = pai_to_idx(trigger_event["pai"])
                    for action_name in self._find_all_chi_kinds(tile_idx):
                        legal_scores[action_name] = score
                continue

            action_name = self._call_action_name_from_decision_and_event(cls_idx, trigger_event)
            if action_name is None:
                continue

            if action_name not in legal_scores or score > legal_scores[action_name]:
                legal_scores[action_name] = score

        return legal_scores

    def _call_legal_scores_from_probs(self, probs, trigger_event, legal_mask):
        probs = probs.detach().cpu().view(-1)
        legal_mask = legal_mask.detach().cpu().view(-1)
        legal_scores = {}

        for cls_idx in range(len(CALL_CLASS_NAMES)):
            if not bool(legal_mask[cls_idx].item()):
                continue

            score = float(probs[cls_idx].item())

            if cls_idx == 1:  # chi
                if trigger_event["type"] == "dahai" and trigger_event["actor"] != self.player_id:
                    tile_idx = pai_to_idx(trigger_event["pai"])
                    for action_name in self._find_all_chi_kinds(tile_idx):
                        legal_scores[action_name] = score
                continue

            action_name = self._call_action_name_from_decision_and_event(cls_idx, trigger_event)
            if action_name is None:
                continue

            if action_name not in legal_scores or score > legal_scores[action_name]:
                legal_scores[action_name] = score

        return legal_scores

    # ----------------------------------------------------------
    # Prediction helpers
    # ----------------------------------------------------------
    @torch.no_grad()
    def _predict_discard(self) -> Tuple[int, dict]:
        x, hist, hist_mask = self._get_state_tensors()

        t0 = time.perf_counter_ns()
        logits = self._forward_discard(x, hist, hist_mask)[0]
        eval_time_ns = time.perf_counter_ns() - t0

        hand_mask = self.round_state.legal_discard_mask(self.player_id)
        masked_logits = self._masked_prediction(logits, hand_mask)
        pred_idx = int(torch.argmax(masked_logits).item())

        legal_scores = self._hand_discard_actions_with_scores(logits, hand_mask)
        meta = self._build_meta_from_legal_scores(legal_scores, eval_time_ns=eval_time_ns, is_greedy=True)
        return pred_idx, meta

    @torch.no_grad()
    def _predict_call(self, called_tile, trigger_event: dict, legal_mask) -> Tuple[int, dict]:
        """
        All call-like decisions come from forward_call:
        pass / chi / pon / hora / dmk / ank / kak / rii
        """
        x, hist, hist_mask = self._get_state_tensors(called_tile=called_tile)

        t0 = time.perf_counter_ns()
        logits = self._forward_call(x, hist, hist_mask)
        eval_time_ns = time.perf_counter_ns() - t0

        
        probs = self._masked_call_prediction(logits, legal_mask)
        decision = int(torch.argmax(probs, dim=1).item())

        legal_scores = self._call_legal_scores(logits[0], trigger_event, legal_mask[0])
        meta = self._build_meta_from_legal_scores(legal_scores, eval_time_ns=eval_time_ns, is_greedy=True)
        return decision, meta

    # ----------------------------------------------------------
    # Tile / meld helpers
    # ----------------------------------------------------------
    def _can_pon(self, tile_idx: int) -> bool:
        return self.round_state.hand_counts(self.player_id)[tile_idx] >= 2

    def _can_chi(self, tile_idx: int, discarder: int) -> bool:
        if (discarder + 1) % 4 != self.player_id:
            return False
        if tile_idx >= 27:
            return False

        hand_cnts = self.round_state.hand_counts(self.player_id)
        suit_start = (tile_idx // 9) * 9
        pos = tile_idx - suit_start

        if pos >= 2 and hand_cnts[suit_start + pos - 2] > 0 and hand_cnts[suit_start + pos - 1] > 0:
            return True
        if 1 <= pos <= 7 and hand_cnts[suit_start + pos - 1] > 0 and hand_cnts[suit_start + pos + 1] > 0:
            return True
        if pos <= 6 and hand_cnts[suit_start + pos + 1] > 0 and hand_cnts[suit_start + pos + 2] > 0:
            return True
        return False
    
    def _find_all_chi_kinds(self, tile_idx: int) -> List[str]:
        if tile_idx >= 27:
            return []

        hand_cnts = self.round_state.hand_counts(self.player_id)
        suit_start = (tile_idx // 9) * 9
        pos = tile_idx - suit_start
        out = []

        if pos >= 2:
            a, b = suit_start + pos - 2, suit_start + pos - 1
            if hand_cnts[a] > 0 and hand_cnts[b] > 0:
                out.append("chi_low")

        if 1 <= pos <= 7:
            a, b = suit_start + pos - 1, suit_start + pos + 1
            if hand_cnts[a] > 0 and hand_cnts[b] > 0:
                out.append("chi_mid")

        if pos <= 6:
            a, b = suit_start + pos + 1, suit_start + pos + 2
            if hand_cnts[a] > 0 and hand_cnts[b] > 0:
                out.append("chi_high")

        return out

    def _find_chi_kind(self, tile_idx: int) -> Optional[str]:
        """
        Akagi naming:
        - chi_low:  called tile is the high tile in the sequence (x-2, x-1, x)
        - chi_mid:  called tile is the middle tile (x-1, x, x+1)
        - chi_high: called tile is the low tile in the sequence (x, x+1, x+2)

        We keep the same search order as _find_chi_consumed().
        """
        if tile_idx >= 27:
            return None

        hand_cnts = self.round_state.hand_counts(self.player_id)
        suit_start = (tile_idx // 9) * 9
        pos = tile_idx - suit_start

        if pos >= 2:
            a, b = suit_start + pos - 2, suit_start + pos - 1
            if hand_cnts[a] > 0 and hand_cnts[b] > 0:
                return "chi_low"

        if 1 <= pos <= 7:
            a, b = suit_start + pos - 1, suit_start + pos + 1
            if hand_cnts[a] > 0 and hand_cnts[b] > 0:
                return "chi_mid"

        if pos <= 6:
            a, b = suit_start + pos + 1, suit_start + pos + 2
            if hand_cnts[a] > 0 and hand_cnts[b] > 0:
                return "chi_high"

        return None

    def _find_chi_consumed(self, tile_idx: int):
        if tile_idx >= 27:
            return None

        hand_cnts = self.round_state.hand_counts(self.player_id)
        suit_start = (tile_idx // 9) * 9
        pos = tile_idx - suit_start
        sequences = []

        if pos >= 2:
            sequences.append((suit_start + pos - 2, suit_start + pos - 1))
        if 1 <= pos <= 7:
            sequences.append((suit_start + pos - 1, suit_start + pos + 1))
        if pos <= 6:
            sequences.append((suit_start + pos + 1, suit_start + pos + 2))

        for a, b in sequences:
            if hand_cnts[a] > 0 and hand_cnts[b] > 0:
                return [idx_to_pai(a), idx_to_pai(b)]
        return None

    def _find_pon_consumed(self, tile_idx: int):
        found = []
        for pai in self.round_state.hands[self.player_id]:
            if pai_to_idx(pai) == tile_idx:
                found.append(pai)
                if len(found) == 2:
                    break
        return found if len(found) == 2 else None

    def _find_daiminkan_consumed(self, tile_idx: int):
        found = []
        for pai in self.round_state.hands[self.player_id]:
            if pai_to_idx(pai) == tile_idx:
                found.append(pai)
                if len(found) == 3:
                    break
        return found if len(found) == 3 else None

    def _find_ankan_consumed(self):
        hand = self.round_state.hands[self.player_id]
        counts = self.round_state.hand_counts(self.player_id)
        for tile_idx in range(34):
            if counts[tile_idx] >= 4:
                found = []
                for pai in hand:
                    if pai_to_idx(pai) == tile_idx:
                        found.append(pai)
                        if len(found) == 4:
                            return found
        return None

    def _find_kakan_pai(self):
        if not hasattr(self.round_state, "melds"):
            return None

        hand = self.round_state.hands[self.player_id]
        counts = self.round_state.hand_counts(self.player_id)
        player_melds = self.round_state.melds[self.player_id]

        for meld in player_melds:
            meld_pais = meld.get("pais") or meld.get("consumed") or []
            if len(meld_pais) < 3:
                continue

            tile_idx = pai_to_idx(meld_pais[0])
            if any(pai_to_idx(p) != tile_idx for p in meld_pais[:3]):
                continue

            if counts[tile_idx] > 0:
                for pai in hand:
                    if pai_to_idx(pai) == tile_idx:
                        return pai
        return None

    # ----------------------------------------------------------
    # Action construction from call class
    # ----------------------------------------------------------
    def _build_action_from_call_decision(self, decision: int, trigger_event: dict) -> Optional[dict]:
        rs = self.round_state
        etype = trigger_event["type"]

        if decision == 0:
            return None

        if etype == "dahai" and trigger_event["actor"] != self.player_id:
            discarder = trigger_event["actor"]
            pai = trigger_event["pai"]
            tile_idx = pai_to_idx(pai)

            if decision == 3:
                return {
                    "type": "hora",
                    "actor": self.player_id,
                    "target": discarder,
                    "pai": pai,
                }

            if decision == 2:
                consumed = self._find_pon_consumed(tile_idx)
                if consumed is not None:
                    return {
                        "type": "pon",
                        "actor": self.player_id,
                        "target": discarder,
                        "pai": pai,
                        "consumed": consumed,
                    }
                return None

            if decision == 1:
                consumed = self._find_chi_consumed(tile_idx)
                if consumed is not None:
                    return {
                        "type": "chi",
                        "actor": self.player_id,
                        "target": discarder,
                        "pai": pai,
                        "consumed": consumed,
                    }
                return None

            if decision == 4:
                consumed = self._find_daiminkan_consumed(tile_idx)
                if consumed is not None:
                    return {
                        "type": "daiminkan",
                        "actor": self.player_id,
                        "target": discarder,
                        "pai": pai,
                        "consumed": consumed,
                    }
                return None

            return None

        if etype == "tsumo" and trigger_event["actor"] == self.player_id:
            pai = trigger_event["pai"]

            if decision == 3:
                return {
                    "type": "hora",
                    "actor": self.player_id,
                    "target": self.player_id,
                    "pai": pai,
                }

            if decision == 7:
                riichi_discards = rs.find_riichi_discards(self.player_id)
                if not riichi_discards:
                    return None

                idx, discard_meta = self._predict_discard()
                if idx not in riichi_discards:
                    idx = riichi_discards[0]

                discard_pai = rs.choose_discard_tile(self.player_id, idx)
                return {
                    "type": "reach",
                    "actor": self.player_id,
                    "pai": discard_pai,
                    "meta": discard_meta,
                }

            if decision == 5:
                consumed = self._find_ankan_consumed()
                if consumed is not None:
                    return {
                        "type": "ankan",
                        "actor": self.player_id,
                        "consumed": consumed,
                    }
                return None

            if decision == 6:
                pai_to_add = self._find_kakan_pai()
                if pai_to_add is not None:
                    return {
                        "type": "kakan",
                        "actor": self.player_id,
                        "pai": pai_to_add,
                    }
                return None

            return None

        return None

    # ----------------------------------------------------------
    # Main decision logic
    # ----------------------------------------------------------
    def _maybe_act(self, last_event: dict) -> Optional[dict]:
        etype = last_event["type"]
        rs = self.round_state

        if etype == "dahai" and last_event["actor"] != self.player_id:
            pai = last_event["pai"]
            legal_mask = self.round_state.legal_call_mask_from_event(self.player_id, last_event).unsqueeze(0)
            decision, call_meta = self._predict_call(pai, last_event, legal_mask)
            action = self._build_action_from_call_decision(decision, last_event)

            if action is not None:
                action["meta"] = call_meta
                return action

            # Return pass-with-meta so UI can show recommendations
            if action is None and legal_mask.sum() > 1:
                return {
                    "type": "none",
                    "meta": call_meta,
                }

        # ------------------------------------------------------
        # Own tsumo: build a unified recommendation space
        # (call actions like reach/hora/kan + discard actions)
        # Skip call recommendations entirely if only pass is legal.
        # ------------------------------------------------------
        if etype == "tsumo" and last_event["actor"] == self.player_id:
            pai = last_event["pai"]

            # ----- call-side inference -----
            legal_mask = self.round_state.legal_call_mask_from_event(
                self.player_id, last_event
            ).unsqueeze(0)

            x_call, hist_call, hist_mask_call = self._get_state_tensors(called_tile=pai)

            t0 = time.perf_counter_ns()
            call_logits = self._forward_call(x_call, hist_call, hist_mask_call)
            call_eval_ns = time.perf_counter_ns() - t0

            call_probs = self._masked_call_prediction(call_logits, legal_mask)
            call_decision = int(torch.argmax(call_probs, dim=1).item())

            # Only include call recommendations if something other than pass is legal
            has_real_call_option = bool(legal_mask[0, 1:].any().item())
            call_scores = {}
            if has_real_call_option:
                call_probs = self._masked_call_prediction(call_logits, legal_mask)
                call_scores = self._call_legal_scores_from_probs(
                    call_probs[0], last_event, legal_mask[0]
                )

            # ----- discard-side inference -----
            x_discard, hist_discard, hist_mask_discard = self._get_state_tensors()

            t1 = time.perf_counter_ns()
            discard_logits = self._forward_discard(
                x_discard, hist_discard, hist_mask_discard
            )[0]
            discard_eval_ns = time.perf_counter_ns() - t1

            hand_mask = self.round_state.legal_discard_mask(self.player_id)
            masked_discard_logits = self._masked_prediction(discard_logits, hand_mask)
            discard_idx = int(torch.argmax(masked_discard_logits).item())

            discard_scores = self._hand_discard_actions_with_scores(
                discard_logits, hand_mask
            )

            # ----- unified meta for UI -----
            merged_scores = {}
            merged_scores.update(discard_scores)
            if has_real_call_option:
                merged_scores.update(call_scores)

            unified_meta = self._build_meta_from_legal_scores(
                merged_scores,
                eval_time_ns=call_eval_ns + discard_eval_ns,
                is_greedy=True,
            )

            # ----- execute chosen call action first if model wants one -----
            action = self._build_action_from_call_decision(call_decision, last_event)
            if action is not None:
                action["meta"] = unified_meta
                return action

            if rs.riichi[self.player_id]:
                return {
                    "type": "skip",
                    "actor": self.player_id,
                    "meta": unified_meta,
                }

            discard_pai = rs.choose_discard_tile(self.player_id, discard_idx)
            return {
                "type": "dahai",
                "actor": self.player_id,
                "pai": discard_pai,
                "tsumogiri": (rs.last_draw[self.player_id] == discard_pai),
                "meta": unified_meta,
            }

        # ------------------------------------------------------
        # After our chi / pon, just discard
        # ------------------------------------------------------
        if etype in {"chi", "pon"} and last_event["actor"] == self.player_id:
            idx, discard_meta = self._predict_discard()
            pai = rs.choose_discard_tile(self.player_id, idx)
            return {
                "type": "dahai",
                "actor": self.player_id,
                "pai": pai,
                "tsumogiri": False,
                "meta": discard_meta,
            }

        return None

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------
    def react(self, events: str) -> str:
        try:
            payload = json.loads(events)
            if isinstance(payload, dict):
                events = [payload]
            elif isinstance(payload, list):
                events = payload
            else:
                raise ValueError(f"Unexpected payload type: {type(payload)}")
        except json.JSONDecodeError as e:
            print(f"Failed to parse events: {events}, {e}", file=sys.stderr)
            return json.dumps({"type": "none"}, separators=(",", ":"))

        return_action = None

        for e in events:
            t = e["type"]

            if t == "start_game":
                self.player_id = e["id"]
                self.round_state = RoundState()
                self._load_model()
                return_action = {"type": "none"}
                continue

            if t == "end_game":
                self.round_state = None
                self._unload_model()
                return_action = {"type": "none"}
                continue

            if self.player_id is None or self.round_state is None:
                continue

            # Opponent discard: react before applying event.
            if t == "dahai" and e["actor"] != self.player_id:
                maybe = self._maybe_act(e)
                if maybe is not None:
                    return_action = maybe

            self.round_state.apply_event(e)

            # Own tsumo / chi / pon: react after applying event.
            if t in {"tsumo", "chi", "pon"} and e.get("actor") == self.player_id:
                maybe = self._maybe_act(e)
                if maybe is not None:
                    return_action = maybe

        if return_action is None:
            return json.dumps({"type": "none"}, separators=(",", ":"))
        return json.dumps(return_action, separators=(",", ":"))

def main():
    bot = Bot()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            res = bot.react(line)
        except Exception as e:
            print(f"Bot error: {e}", file=sys.stderr)
            res = json.dumps({"type": "none"}, separators=(",", ":"))
        print(res, flush=True)


if __name__ == "__main__":
    main()
