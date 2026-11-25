import shutil
import subprocess
import time

class AudioAlarm:
    PLAYERS = (("paplay", []), ("pw-play", []), ("aplay", ["-q"]), ("ffplay", ["-nodisp", "-autoexit", "-loglevel", "quiet"]))

    def __init__(self, wav_path: str = "jarvis_alarm.wav", cooldown_seconds: float = 1.8):
        self.wav_path = wav_path
        self.cooldown_seconds = cooldown_seconds
        self._muted = False
        self._last_trigger = 0.0
        self._player_cmd = next(([shutil.which(cmd)] + args for cmd, args in self.PLAYERS if shutil.which(cmd)), None)
        
    def mute(self) -> None:
        self._muted = True

    def trigger(self, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        if self._muted or (now - self._last_trigger) < self.cooldown_seconds:
            return False
        
        self._last_trigger = now

        if self._player_cmd is not None:
            try:
                subprocess.Popen(self._player_cmd + [self.wav_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
            except Exception:
                pass
            
        print("\a", end="", flush=True)
        return True