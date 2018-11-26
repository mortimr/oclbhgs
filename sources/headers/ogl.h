//
// Created by Iulian Rotaru on 2018-11-27.
//

#ifndef OCLBHGS_OGL_H
#define OCLBHGS_OGL_H

#ifdef __APPLE__

#include <GLUT/glut.h>
#include <z3.h>

#else
#include <GL/glut.h>
#endif

#include "body.h"
#include "keys.h"

#define WINDOW_WIDTH (1000)
#define WINDOW_HEIGHT (1000)

typedef struct ogl {
    int window;
    float sim_x;
    float sim_y;
    float ratio_x;
    float ratio_y;
    keys *keys;
} ogl;

void ogl_init(void (*display)(),
              int *argc,
              char **argv,
              float sim_x,
              float sim_y,
              keys *keys_data);

void ogl_draw_bodies(body *bodies, size_t body_count);

void ogl_draw_quadrants(cell *cells, size_t cell_count);


#endif //OCLBHGS_OGL_H
