from autoplay.majsoul.majsoul_autoplay import MajsoulAutoPlay

from .logger import logger
from settings.settings import settings, MITMType
import win32gui
import win32con
from .window import WindowObject
    
class AutoPlay(object):
    def __init__(self):
        self.bot = None
        self.autoplay_instance = None
        self._target_window: WindowObject = None
        
    @property
    def target_window(self) -> WindowObject:
        """
        Returns the target window object.
        The target window is the first visible window in the list of windows.
        """
        return self._target_window

    def set_bot(self, bot):
        """
        Args:
            bot (AkagiBot): The AkagiBot instance to be used.

        Returns:
            None: No return value.
        """
        self.bot = bot

    def set_autoplay(self):
        """
        Args:
            autoplay (AutoPlayBase): The AutoPlayBase instance to be used.

        Returns:
            None: No return value.
        """
        match settings.mitm.type:
            case MITMType.AMATSUKI:
                return
            case MITMType.MAJSOUL:
                self.autoplay_instance = MajsoulAutoPlay()
            case MITMType.RIICHI_CITY:
                return
            case MITMType.TENHOU:
                return
            case MITMType.UNIFIED:
                return
            case _:
                logger.error(f"Unknown MITM type: {settings.mitm.type}")
                return

    def get_windows(self) -> list[WindowObject]:
        """
        Returns a list of WindowObject instances for all visible windows.
        Each WindowObject contains the window handle (hwnd) and window name.
        """
        windows = []
        def enum_callback(hwnd, _):
            if win32gui.IsWindowVisible(hwnd):
                name = win32gui.GetWindowText(hwnd)
                if name:  # skip empty titles
                    windows.append(WindowObject(hwnd, name))

        win32gui.EnumWindows(enum_callback, None)
        return windows
    
    def select_window(self, hwnd: int) -> None:
        """
        Selects a window by its handle (hwnd).
        """
        # restore if minimized
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        
        # bring to foreground
        win32gui.SetForegroundWindow(hwnd)

        name = win32gui.GetWindowText(hwnd)
        self._target_window = WindowObject(hwnd, name)
    
    def check_window(self) -> bool:
        """
        Checks if the target window is valid and visible.
        Returns True if the target window is valid, False otherwise.
        """
        if self.autoplay_instance is None:
            return False
        
        if self.target_window is None:
            return False
        
        # window must be visible
        if not win32gui.IsWindowVisible(self.target_window.hwnd):
            return False

        # window must not be minimized
        if win32gui.IsIconic(self.target_window.hwnd):
            return False

        return self.autoplay_instance.check_window(self._target_window)
    
    def auto_select_window(self) -> WindowObject:
        """
        Automatically selects the window based on the current settings.
        """
        if self.autoplay_instance is None:
            return
        self._target_window = self.autoplay_instance.auto_select_window(self.get_windows())
        return self.target_window

    def act(self, mjai_msg: dict) -> bool:
        """
        Given a MJAI message, this method processes the message and performs the corresponding action.

        Args:
            mjai_msg (dict): The MJAI message to process.

        Returns:
            bool: True if the action was performed, False otherwise.
        """
        if self.autoplay_instance is None:
            return False
        if not self.check_window():
            self.auto_select_window()
            if not self.check_window():
                return False
        if mjai_msg["type"] == "skip":
            return True
        elif mjai_msg["type"] == "dahai":
            return self.autoplay_instance.click_discard(self.target_window, mjai_msg)
        elif mjai_msg["type"] in ["chi", "pon", "none", "daiminkan", "kakan", "ankan", "reach", "hora"]:
            return self.autoplay_instance.click_action(self.target_window, mjai_msg)

        return False
