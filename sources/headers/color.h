//
// Created by Iulian Rotaru on 2018-12-01.
//

#ifndef OCLBHGS_COLOR_H
#define OCLBHGS_COLOR_H

typedef enum color_target {
   BODY = 0,
   QUADRANT
} color_target;

typedef struct color {
    float r;
    float g;
    float b;
} color;

#endif //OCLBHGS_COLOR_H
