#pragma once

// IMPORTANT: Keep in sync with ../signal_kernel_ids.py
// Char-based IDs avoid ABI/enum-size issues across Python<->C boundaries.

#define SIGK_IDENTITY  'I'
#define SIGK_DEADZONE  'D'
#define SIGK_CLAMP     'C'
#define SIGK_SMOOTHSTEP 'S'
#define SIGK_EXPO      'E'
