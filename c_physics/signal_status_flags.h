#pragma once

// IMPORTANT:
// - Keep in sync with ../signal_status_flags.py
// - These flags describe *status* about a signal at an emitted frame.

#include <stdint.h>

#define SIGF_DOWN         (1u << 0)  // signal is currently asserted/active/down
#define SIGF_HOLD         (1u << 1)  // asserted long enough to count as hold
#define SIGF_DOWN_EDGE    (1u << 2)  // transitioned up->down this frame
#define SIGF_UP_EDGE      (1u << 3)  // transitioned down->up this frame
#define SIGF_DOUBLE_EDGE  (1u << 4)  // down edge is a "double" (2nd press within window)
#define SIGF_DOUBLE_HOLD  (1u << 5)  // double press + held long enough

#define SIGF_TOGGLED            (1u << 6)  // latched state is ON
#define SIGF_TOGGLE_EDGE        (1u << 7)  // latched state changed this frame
#define SIGF_DOUBLE_TOGGLE_EDGE (1u << 8)  // toggle edge caused by a double-press
