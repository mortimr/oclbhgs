//
// Created by Iulian Rotaru on 2018-11-28.
//

#ifndef OCLBHGS_KEYS_H
#define OCLBHGS_KEYS_H

#ifdef __APPLE__

#else

# include <stdbool.h>

#endif

typedef struct keys {
    bool paused;
    bool grid;
    float fps;
} keys;

#endif //OCLBHGS_KEYS_H
