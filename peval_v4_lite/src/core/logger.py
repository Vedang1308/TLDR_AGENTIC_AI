class PEVLogger:
    """
    Centralized logging utility for clean, understandable terminal output.
    Uses ANSI color codes to distinguish components without overwhelming the screen.
    """
    # ANSI Colors
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

    @staticmethod
    def step(step_num: int):
        print(f"\n{PEVLogger.HEADER}{PEVLogger.BOLD}=== STEP {step_num} ==={PEVLogger.RESET}")

    @staticmethod
    def node(node_name: str, action: str):
        print(f"{PEVLogger.CYAN}[{node_name}]{PEVLogger.RESET} {action}")

    @staticmethod
    def info(msg: str):
        # Truncate very long messages to keep terminal clean
        if len(msg) > 150:
            msg = msg[:147] + "..."
        print(f"  {PEVLogger.BLUE}↳{PEVLogger.RESET} {msg}")

    @staticmethod
    def success(msg: str):
        print(f"  {PEVLogger.GREEN}✓{PEVLogger.RESET} {msg}")

    @staticmethod
    def warn(msg: str):
        print(f"  {PEVLogger.WARNING}⚠{PEVLogger.RESET} {msg}")

    @staticmethod
    def error(msg: str):
        print(f"  {PEVLogger.FAIL}✗{PEVLogger.RESET} {msg}")
