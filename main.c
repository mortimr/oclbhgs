#include <stdio.h>
#include "sources/headers/galaxy.h"
#include "sources/headers/ocl.h"
#include "sources/headers/body.h"
#include "sources/headers/ogl.h"
#include <time.h>
#include <math.h>
#include <stdlib.h>

#ifdef __APPLE__

#include <GLUT/glut.h>
#include <zconf.h>

#else

#include <GL/glut.h>
#include <unistd.h>

#endif

#define MAP_WIDTH (2048)
#define MAP_HEIGHT (2048)
#define GALAXY_ONE_BODY_COUNT (1001)
#define GALAXY_ONE_THETA_VALUE (0.8)
#define GALAXY_ONE_G_VALUE (0.00009674)
#define GALAXY_ONE_MAX_DEPTH (7)

keys KEYS;

void loop() {
    static ocl ocl;
    static body *bodies_one = NULL;
    static cell *cells_one = NULL;
    static ocl_galaxy *galaxy_one = NULL;

    if (!ocl.ctx) {
        ocl_init(&ocl);
    }

    if (!bodies_one) {
        srand((unsigned int) time(NULL));
        bodies_one = body_init_set(GALAXY_ONE_BODY_COUNT);

        size_t global_idx = 0;

        for (size_t body_idx = 0; body_idx < GALAXY_ONE_BODY_COUNT - 1; ++body_idx) {
            float x = rand() % MAP_WIDTH / 4 + MAP_WIDTH / 8;
            float y = rand() % MAP_HEIGHT / 4 + MAP_HEIGHT / 8;
            float dist = sqrt(pow((float) (x - MAP_WIDTH / 4), (float) 2) +
                              pow((float) (y - MAP_HEIGHT / 4), (float) 2));
            if (dist > MAP_WIDTH / 8) {
                --body_idx;
                continue ;
            }

            bodies_one[body_idx].pos.x = x;
            bodies_one[body_idx].pos.y = y;
            bodies_one[body_idx].speed.x = 3;
            bodies_one[body_idx].speed.y = -1;
            bodies_one[global_idx + body_idx].mass = 0;
        }
        bodies_one[GALAXY_ONE_BODY_COUNT - 1].pos.x = MAP_WIDTH / 2;
        bodies_one[GALAXY_ONE_BODY_COUNT - 1].pos.y = MAP_HEIGHT / 2;
        bodies_one[GALAXY_ONE_BODY_COUNT - 1].mass = 150000000;

    }

    if (!galaxy_one) {
        galaxy_one = galaxy_init(&ocl, &cells_one, GALAXY_ONE_MAX_DEPTH, GALAXY_ONE_THETA_VALUE, GALAXY_ONE_G_VALUE, bodies_one,
                                 GALAXY_ONE_BODY_COUNT, MAP_WIDTH, MAP_HEIGHT);
    }

    struct timespec start_spec;
    clock_gettime(CLOCK_MONOTONIC, &start_spec);

    if (!KEYS.paused) {
        galaxy_resolve(galaxy_one, &ocl);
        galaxy_compute(galaxy_one, &ocl);
    }

    galaxy_recover_bodies(galaxy_one, &ocl, bodies_one);
    galaxy_recover_cells(galaxy_one, &ocl, cells_one);

    glClear(GL_COLOR_BUFFER_BIT);
    glLineWidth(1);
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    if (KEYS.grid) {
        glColor3f(0, 0.3, 0.3);
        ogl_draw_quadrants(cells_one, galaxy_one->cell_count);
        glColor3f(1, 1, 1);
        glRasterPos2i(5, 5);

        char fps[50];

        snprintf(fps, 50, "%g", KEYS.fps);

        for (size_t idx = 0; fps[idx] != 0; ++idx) {
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_10, fps[idx]);
        }
    }

    glBegin(GL_POINTS);
    glLineWidth(1);
    glColor3f(0.6, 0.6, 1);

    ogl_draw_bodies(bodies_one, GALAXY_ONE_BODY_COUNT);

    glEnd();

    glutSwapBuffers();

    struct timespec end_spec;
    clock_gettime(CLOCK_MONOTONIC, &end_spec);

    unsigned long start = ((unsigned long) (start_spec.tv_sec * 1000) + start_spec.tv_nsec / 1000000);
    unsigned long end = ((unsigned long) (end_spec.tv_sec * 1000) + end_spec.tv_nsec / 1000000);

    if (end - start < 1000.0f / KEYS.fps) {
        usleep((useconds_t) ((1000.0f / KEYS.fps) - (end - start)) * 1000);
    }

    glutPostRedisplay();
}

int main(int argc, char *argv[]) {
    KEYS.paused = true;
    KEYS.fps = 60.0f;
    ogl_init(&loop, &argc, argv, MAP_WIDTH, MAP_HEIGHT, &KEYS);
    return 0;
}