class WindowObject:
    def __init__(self, hwnd, name):
        self.hwnd = hwnd
        self.name = name

    def __repr__(self):
        return f"WindowObject(hwnd={self.hwnd}, name='{self.name}')"
