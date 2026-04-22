
import json
import pathlib
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from .logger import logger

try:
    from .model import MahjongDecisionNet
    from .gamestate import (
        RoundState,
        pai_to_idx,
        idx_to_pai,
        is_red_pai,
        tile37_to_base34,
        NUM_TILES,
        DAHAI_ACTION_NAMES,
        TSUMO_ACTION_NAMES,
        CALL_KIND_TO_IDX,
        TSUMO_ACTION_TO_IDX,
    )
except ImportError:
    from model import MahjongDecisionNet
    from gamestate import (
        RoundState,
        pai_to_idx,
        idx_to_pai,
        is_red_pai,
        tile37_to_base34,
        NUM_TILES,
        DAHAI_ACTION_NAMES,
        TSUMO_ACTION_NAMES,
        CALL_KIND_TO_IDX,
        TSUMO_ACTION_TO_IDX,
    )


DAHAI_CONF_THRESHOLDS = {
    1: 0.75,  # chi_low
    2: 0.75,  # chi_mid
    3: 0.75,  # chi_high
    4: 0.73,  # pon
    5: 0.90,  # kan / daiminkan
}

ACTION_ORDER_4P = [
    "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
    "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
    "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
    "E", "S", "W", "N", "P", "F", "C",
    "5mr", "5pr", "5sr",
    "reach", "chi_low", "chi_mid", "chi_high", "pon", "kan_select", "hora", "ryukyoku", "none",
]
ACTION_TO_IDX_4P = {a: i for i, a in enumerate(ACTION_ORDER_4P)}


class Bot:
    def __init__(self, device: Optional[str] = None):
        self.player_id: Optional[int] = None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[MahjongDecisionNet] = None
        self.round_state: Optional[RoundState] = None
        self.model_paths = {
            "dahai": "./best_dahai.pt",
            "tsumo": "./best_tsumo.pt",
        }

    # ----------------------------------------------------------
    # Model loading
    # ----------------------------------------------------------
    def _resolve_model_path(self, rel_path: str) -> pathlib.Path:
        p = pathlib.Path(rel_path)
        if p.exists():
            return p
        alt = pathlib.Path(__file__).parent / rel_path
        if alt.exists():
            return alt
        raise FileNotFoundError(f"Model not found: {rel_path}")

    def _load_model(self):
        if self.model is not None:
            return

        dahai_path = self._resolve_model_path(self.model_paths["dahai"])
        tsumo_path = self._resolve_model_path(self.model_paths["tsumo"])

        model = MahjongDecisionNet().to(self.device)
        model.dahai_model.load_state_dict(torch.load(dahai_path, map_location=self.device))
        model.tsumo_model.load_state_dict(torch.load(tsumo_path, map_location=self.device))
        model.eval()
        self.model = model

    def _unload_model(self):
        self.model = None
        if self.device == "cuda":
            torch.cuda.empty_cache()

    # ----------------------------------------------------------
    # Tensor helpers
    # ----------------------------------------------------------
    def _get_state_tensors(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.round_state.to_feature(self.player_id).unsqueeze(0).to(self.device)
        hist, hist_mask = self.round_state.get_history(self.player_id)
        hist = hist.unsqueeze(0).to(self.device)
        hist_mask = hist_mask.unsqueeze(0).to(self.device)
        return x, hist, hist_mask

    @staticmethod
    def _masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return logits.masked_fill(~mask, -1e9)

    def _masked_softmax(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        masked = self._masked_logits(logits, mask)
        probs = F.softmax(masked, dim=-1)
        probs = probs * mask.float()
        denom = probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return probs / denom

    def _threshold_dahai_probs(self, probs: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
        probs = probs.clone()
        for cls_idx, threshold in DAHAI_CONF_THRESHOLDS.items():
            if cls_idx >= probs.size(-1):
                continue
            probs[:, cls_idx] = torch.where(
                probs[:, cls_idx] >= threshold,
                probs[:, cls_idx],
                torch.zeros_like(probs[:, cls_idx]),
            )

        probs = probs * legal_mask.float()
        probs[:, 0] = torch.where(
            legal_mask[:, 0],
            probs[:, 0],
            torch.zeros_like(probs[:, 0]),
        )
        denom = probs.sum(dim=-1, keepdim=True)
        normalized = probs / denom.clamp_min(1e-12)
        fallback = torch.zeros_like(probs)
        fallback[:, 0] = legal_mask[:, 0].float()
        probs = torch.where(denom > 0, normalized, fallback)
        return probs

    @torch.no_grad()
    def _forward_dahai(self) -> Tuple[torch.Tensor, int]:
        x, hist, hist_mask = self._get_state_tensors()
        t0 = time.perf_counter_ns()
        logits, _ = self.model.forward_dahai(x, hist, hist_mask)
        eval_time_ns = time.perf_counter_ns() - t0
        return logits, eval_time_ns

    @torch.no_grad()
    def _forward_tsumo(self) -> Tuple[torch.Tensor, torch.Tensor, int]:
        x, hist, hist_mask = self._get_state_tensors()
        t0 = time.perf_counter_ns()
        action_logits, tile_logits, _ = self.model.forward_tsumo(x, hist, hist_mask)
        eval_time_ns = time.perf_counter_ns() - t0
        return action_logits, tile_logits, eval_time_ns

    # ----------------------------------------------------------
    # Meta helpers
    # ----------------------------------------------------------
    @staticmethod
    def _keep_topk_ui_probs(ui_probs: Dict[str, float], top_k: int = 3) -> Dict[str, float]:
        if top_k <= 0 or not ui_probs:
            return {}

        ranked = sorted(
            ui_probs.items(),
            key=lambda kv: (-float(kv[1]), ACTION_TO_IDX_4P.get(kv[0], 10**9)),
        )
        kept = ranked[:top_k]
        total = sum(float(prob) for _, prob in kept)

        if total <= 0.0:
            return {name: 0.0 for name, _ in kept}

        return {
            name: float(prob) / total
            for name, prob in kept
        }

    @staticmethod
    def _build_meta_from_ui_probs(
        ui_probs: Dict[str, float],
        eval_time_ns: int,
        is_greedy: bool = True,
        top_k: int = 3,
    ) -> dict:
        ui_probs = Bot._keep_topk_ui_probs(ui_probs, top_k=top_k)

        mask_bits = 0
        q_values: List[float] = []

        for idx, action_name in enumerate(ACTION_ORDER_4P):
            if action_name in ui_probs:
                mask_bits |= (1 << idx)
                q_values.append(float(ui_probs[action_name]))

        return {
            "q_values": q_values,
            "mask_bits": int(mask_bits),
            "is_greedy": bool(is_greedy),
            "eval_time_ns": int(eval_time_ns),
        }

    def _ui_name_for_tile37(self, idx37: int) -> str:
        if idx37 == 34:
            return "5mr"
        if idx37 == 35:
            return "5pr"
        if idx37 == 36:
            return "5sr"
        return idx_to_pai(idx37)

    def _add_tile_distribution(self, out: Dict[str, float], mass: float, tile_probs: torch.Tensor, tile_mask: torch.Tensor):
        if mass <= 0:
            return
        tile_probs = tile_probs.detach().cpu().view(-1)
        tile_mask = tile_mask.detach().cpu().view(-1)
        for idx37 in range(NUM_TILES):
            if not bool(tile_mask[idx37].item()):
                continue
            action_name = self._ui_name_for_tile37(idx37)
            out[action_name] = out.get(action_name, 0.0) + mass * float(tile_probs[idx37].item())

    def _build_dahai_meta(self, probs: torch.Tensor, legal_mask: torch.Tensor, eval_time_ns: int) -> dict:
        probs = probs.detach().cpu().view(-1)
        legal_mask = legal_mask.detach().cpu().view(-1)
        ui_probs: Dict[str, float] = {}

        for cls_idx, action_name in enumerate(DAHAI_ACTION_NAMES):
            if not bool(legal_mask[cls_idx].item()):
                continue
            score = float(probs[cls_idx].item())
            if cls_idx == CALL_KIND_TO_IDX["kan"]:
                ui_probs["kan_select"] = score
            else:
                ui_probs[action_name] = score

        return self._build_meta_from_ui_probs(ui_probs, eval_time_ns)

    def _build_tsumo_meta(
        self,
        action_probs: torch.Tensor,
        discard_tile_probs: torch.Tensor,
        reach_tile_probs: Optional[torch.Tensor],
        discard_mask: torch.Tensor,
        reach_mask: torch.Tensor,
        action_mask: torch.Tensor,
        eval_time_ns: int,
    ) -> dict:
        action_probs = action_probs.detach().cpu().view(-1)
        action_mask = action_mask.detach().cpu().view(-1)

        ui_probs: Dict[str, float] = {}

        dahai_mass = float(action_probs[TSUMO_ACTION_TO_IDX["dahai"]].item()) if bool(action_mask[TSUMO_ACTION_TO_IDX["dahai"]].item()) else 0.0
        self._add_tile_distribution(ui_probs, dahai_mass, discard_tile_probs, discard_mask)

        if bool(action_mask[TSUMO_ACTION_TO_IDX["reach"]].item()):
            ui_probs["reach"] = float(action_probs[TSUMO_ACTION_TO_IDX["reach"]].item())

        if bool(action_mask[TSUMO_ACTION_TO_IDX["kan"]].item()):
            ui_probs["kan_select"] = float(action_probs[TSUMO_ACTION_TO_IDX["kan"]].item())

        if bool(action_mask[TSUMO_ACTION_TO_IDX["hora"]].item()):
            ui_probs["hora"] = float(action_probs[TSUMO_ACTION_TO_IDX["hora"]].item())

        if bool(action_mask[TSUMO_ACTION_TO_IDX["none"]].item()):
            ui_probs["none"] = float(action_probs[TSUMO_ACTION_TO_IDX["none"]].item())

        return self._build_meta_from_ui_probs(ui_probs, eval_time_ns)

    # ----------------------------------------------------------
    # Tile choice helpers
    # ----------------------------------------------------------
    def _choose_tile_from_mask(self, tile_logits: torch.Tensor, tile_mask: torch.Tensor) -> int:
        masked = self._masked_logits(tile_logits, tile_mask)
        return int(masked.argmax(dim=-1).item())

    def _tile_probs_from_mask(self, tile_logits: torch.Tensor, tile_mask: torch.Tensor) -> torch.Tensor:
        return self._masked_softmax(tile_logits, tile_mask)

    def _find_chi_consumed(self, chi_kind: str, called_pai: str) -> Optional[List[str]]:
        base34 = tile37_to_base34(pai_to_idx(called_pai))
        if base34 >= 27:
            return None

        if chi_kind == "chi_low":
            needed = [base34 + 1, base34 + 2]
        elif chi_kind == "chi_mid":
            needed = [base34 - 1, base34 + 1]
        elif chi_kind == "chi_high":
            needed = [base34 - 2, base34 - 1]
        else:
            return None

        consumed: List[str] = []
        for b in needed:
            picked = self.round_state.pick_tiles_by_base34(self.player_id, b, 1, prefer_red=False)
            if not picked:
                return None
            consumed.append(picked[0])
        return consumed

    def _find_pon_consumed(self, called_pai: str) -> Optional[List[str]]:
        base34 = tile37_to_base34(pai_to_idx(called_pai))
        consumed = self.round_state.pick_tiles_by_base34(self.player_id, base34, 2, prefer_red=False)
        return consumed if len(consumed) == 2 else None

    def _find_daiminkan_consumed(self, called_pai: str) -> Optional[List[str]]:
        base34 = tile37_to_base34(pai_to_idx(called_pai))
        consumed = self.round_state.pick_tiles_by_base34(self.player_id, base34, 3, prefer_red=False)
        return consumed if len(consumed) == 3 else None

    def _build_ankan_or_kakan(self, idx37: int) -> Optional[dict]:
        base34 = tile37_to_base34(idx37)
        counts34 = self.round_state.hand_counts_base34(self.player_id)

        if self.round_state.has_pon_meld(self.player_id, base34) and counts34[base34] >= 1:
            return {
                "type": "kakan",
                "actor": self.player_id,
                "pai": idx_to_pai(idx37),
                "consumed": [idx_to_pai(idx37)],
            }

        if counts34[base34] >= 4:
            consumed = self.round_state.pick_tiles_by_base34(self.player_id, base34, 4, prefer_red=False)
            if len(consumed) == 4:
                return {
                    "type": "ankan",
                    "actor": self.player_id,
                    "consumed": consumed,
                }

        return None

    # ----------------------------------------------------------
    # Prediction / action builders
    # ----------------------------------------------------------
    def _maybe_react_to_dahai(self) -> Optional[dict]:
        mask = self.round_state.legal_dahai_reaction_mask(self.player_id)
        legal_mask = mask.unsqueeze(0)
        if not bool(legal_mask[0, 1:].any().item()):
            return None

        logits, eval_time_ns = self._forward_dahai()
        probs = self._masked_softmax(logits, legal_mask)
        probs = self._threshold_dahai_probs(probs, legal_mask)
        decision = int(probs.argmax(dim=-1).item())

        meta = self._build_dahai_meta(probs[0], legal_mask[0], eval_time_ns)
        last_actor = self.round_state.last_discard_actor
        last_pai = idx_to_pai(self.round_state.last_discard_tile)

        if decision == CALL_KIND_TO_IDX["none"]:
            return {"type": "none", "meta": meta}

        if decision == CALL_KIND_TO_IDX["hora"]:
            return {
                "type": "hora",
                "actor": self.player_id,
                "target": last_actor,
                "pai": last_pai,
                "meta": meta,
            }

        if decision == CALL_KIND_TO_IDX["pon"]:
            consumed = self._find_pon_consumed(last_pai)
            if consumed is None:
                return {"type": "none", "meta": meta}
            return {
                "type": "pon",
                "actor": self.player_id,
                "target": last_actor,
                "pai": last_pai,
                "consumed": consumed,
                "meta": meta,
            }

        if decision == CALL_KIND_TO_IDX["kan"]:
            consumed = self._find_daiminkan_consumed(last_pai)
            if consumed is None:
                return {"type": "none", "meta": meta}
            return {
                "type": "daiminkan",
                "actor": self.player_id,
                "target": last_actor,
                "pai": last_pai,
                "consumed": consumed,
                "meta": meta,
            }

        chi_name = DAHAI_ACTION_NAMES[decision]
        consumed = self._find_chi_consumed(chi_name, last_pai)
        if consumed is None:
            return {"type": "none", "meta": meta}
        legal_chi_count = (1 if mask[CALL_KIND_TO_IDX["chi_low"]] else 0) + (1 if mask[CALL_KIND_TO_IDX["chi_mid"]] else 0) + (1 if mask[CALL_KIND_TO_IDX["chi_high"]] else 0)
        return {
            "type": "chi",
            "actor": self.player_id,
            "target": last_actor,
            "pai": last_pai,
            "consumed": consumed,
            "meta": meta,
            "chi_count": legal_chi_count
        }

    def _maybe_act_on_own_turn(self) -> Optional[dict]:
        masks = self.round_state.legal_tsumo_action_masks(self.player_id)
        action_mask = masks["action_mask"].unsqueeze(0).to(self.device)
        discard_mask = masks["discard_mask"].unsqueeze(0).to(self.device)
        reach_mask = masks["reach_mask"].unsqueeze(0).to(self.device)
        kan_mask = masks["kan_mask"].unsqueeze(0).to(self.device)

        action_logits, tile_logits, eval_time_ns = self._forward_tsumo()
        action_probs = self._masked_softmax(action_logits, action_mask)

        discard_tile_probs = self._tile_probs_from_mask(tile_logits, discard_mask)
        reach_tile_probs = self._tile_probs_from_mask(tile_logits, reach_mask) if bool(reach_mask.any().item()) else None
        kan_tile_probs = self._tile_probs_from_mask(tile_logits, kan_mask) if bool(kan_mask.any().item()) else None

        meta = self._build_tsumo_meta(
            action_probs[0],
            discard_tile_probs[0],
            reach_tile_probs[0] if reach_tile_probs is not None else None,
            discard_mask[0],
            reach_mask[0],
            action_mask[0],
            eval_time_ns,
        )

        decision = int(action_probs.argmax(dim=-1).item())

        if decision == TSUMO_ACTION_TO_IDX["hora"]:
            pai = self.round_state.last_draw[self.player_id]
            if pai is None:
                pai = idx_to_pai(self._choose_tile_from_mask(tile_logits, discard_mask))
            idx37 = self._choose_tile_from_mask(tile_logits, reach_mask)
            return {
                "type": "hora",
                "actor": self.player_id,
                "target": self.player_id,
                "pai": pai,
                "meta": meta,
                "default_pai": pai
            }

        if self.round_state.riichi[self.player_id]:
            pai = self.round_state.last_draw[self.player_id]
            if pai is None:
                idx37 = self._choose_tile_from_mask(tile_logits, discard_mask)
                pai = self.round_state.choose_discard_tile(self.player_id, idx37)
            return {
                "type": "dahai",
                "actor": self.player_id,
                "pai": pai,
                "tsumogiri": True,
                "meta": meta,
                "skip_play": True
            }

        if decision == TSUMO_ACTION_TO_IDX["reach"] and bool(reach_mask.any().item()):
            idx37 = self._choose_tile_from_mask(tile_logits, reach_mask)
            pai = self.round_state.choose_discard_tile(self.player_id, idx37)
            return {
                "type": "reach",
                "actor": self.player_id,
                "pai": pai,
                "meta": meta,
            }

        if decision == TSUMO_ACTION_TO_IDX["kan"] and bool(kan_mask.any().item()):
            idx37 = self._choose_tile_from_mask(tile_logits, kan_mask)
            kan_action = self._build_ankan_or_kakan(idx37)
            if kan_action is not None:
                kan_action["meta"] = meta
                return kan_action

        idx37 = self._choose_tile_from_mask(tile_logits, discard_mask)
        pai = self.round_state.choose_discard_tile(self.player_id, idx37)
        return {
            "type": "dahai",
            "actor": self.player_id,
            "pai": pai,
            "tsumogiri": (self.round_state.last_draw[self.player_id] == pai),
            "meta": meta,
        }

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
                self.round_state = RoundState(player_id=self.player_id)
                self._load_model()
                return_action = {"type": "start_game"}
                continue

            if t == "end_game":
                self.round_state = None
                self._unload_model()
                return_action = {"type": "end_game"}
                continue

            if self.player_id is None or self.round_state is None:
                continue

            self.round_state.apply_event(e)
            
            if t == "reach_accepted" or t == "reach":
                return_action = {"type": "none", "skip_play": True}

            if t == "dahai" and e.get("actor") != self.player_id:
                maybe = self._maybe_react_to_dahai()
                if maybe is not None:
                    return_action = maybe
                continue

            if t in {"tsumo", "chi", "pon"} and e.get("actor") == self.player_id:
                maybe = self._maybe_act_on_own_turn()
                if maybe is not None:
                    return_action = maybe
                    if self.round_state.last_discard_tile == None:
                        return_action["first_action"] = True
                continue

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
