# Controller Timers (C backend)

This repo’s C controller engine can synthesize periodic timer pulses by writing **virtual button-like signals** into the C signal-kernel each tick.

## Config

Add timers under:

- `flight_controls.controller.timers`

Example (1 Hz pulse when `tick_hz=240`):

```json
{
  "flight_controls": {
    "controller": {
      "timers": [
        {"timer_id": 0, "period_ticks": 240, "duty_ticks": 1, "phase_ticks": 0, "repeat": true}
      ]
    }
  }
}
```

You can also use seconds/milliseconds instead of ticks:

```json
{"timer_id": 1, "period_s": 0.5, "duty_ms": 10}
```

## Listening from menus (hooks)

Hook bindings can watch the timer signal like any other signal.

Timer `timer_id` is represented as a **virtual joystick button** with `item_id = 60000 + timer_id`.

Example: fire an action on each pulse start (rising edge):

```json
{
  "flight_controls": {
    "controller": {
      "hook_bindings": [
        {
          "action": "menu_heartbeat",
          "source": "signal",
          "device": 3,
          "kind": 2,
          "item_id": 60000,
          "sigsel": 2,
          "edge": "rise",
          "threshold": 0.5,
          "hysteresis": 0.05,
          "dispatch": "main"
        }
      ]
    }
  }
}
```

Notes:
- `device=3` is `GP_DEV_JOYSTICK`, `kind=2` is `GP_EV_BUTTON`, `sigsel=2` is `GP_SIGSEL_DOWN`.
- You can also watch `sigsel=3` (`GP_SIGSEL_DOWN_EDGE`) and use `edge: "level"` if you prefer edge pulses.
