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

#define GALAXY_COUNT (2)
#define GALAXY_MAX_DEPTH (7)

#define GALAXY_ONE_BODY_COUNT (2751)
#define GALAXY_ONE_THETA_VALUE (0.8)
#define GALAXY_ONE_G_VALUE (0.00006674)

#define GALAXY_TWO_BODY_COUNT (2751)
#define GALAXY_TWO_THETA_VALUE (0.8)
#define GALAXY_TWO_G_VALUE (0.00006674)

keys KEYS;

void loop() {
    static ocl ocl;
    static body *bodies[GALAXY_COUNT];
    static cell *cells[GALAXY_COUNT];
    static ocl_galaxy *galaxies = NULL;

    if (!ocl.ctx) {
        ocl_init(&ocl);
    }

    if (!bodies[0]) {
        srand((unsigned int) time(NULL));
        bodies[0] = body_init_set(GALAXY_ONE_BODY_COUNT);

        size_t global_idx = 0;

        for (size_t body_idx = 0; body_idx < GALAXY_ONE_BODY_COUNT - 1; ++body_idx) {
            float x = rand() % MAP_WIDTH / 4 + MAP_WIDTH / 8 + MAP_WIDTH / 2;
            float y = rand() % MAP_HEIGHT / 4 + MAP_HEIGHT / 8;
            float dist = sqrt(pow((float) (x - (MAP_WIDTH / 4 + MAP_WIDTH / 2)), (float) 2) +
                              pow((float) (y - MAP_HEIGHT / 4), (float) 2));
            if (dist > MAP_WIDTH / 8) {
                --body_idx;
                continue;
            }

            bodies[0][body_idx].pos.x = x;
            bodies[0][body_idx].pos.y = y;
            bodies[0][body_idx].speed.x = 1;
            bodies[0][body_idx].speed.y = 2;
            bodies[0][global_idx + body_idx].mass = 0;
        }
        bodies[0][GALAXY_ONE_BODY_COUNT - 1].pos.x = MAP_WIDTH / 2;
        bodies[0][GALAXY_ONE_BODY_COUNT - 1].pos.y = MAP_HEIGHT / 2;
        bodies[0][GALAXY_ONE_BODY_COUNT - 1].mass = 100000000;

    }

    if (!bodies[1]) {
        srand((unsigned int) time(NULL));
        bodies[1] = body_init_set(GALAXY_TWO_BODY_COUNT);

        size_t global_idx = 0;

        for (size_t body_idx = 0; body_idx < GALAXY_TWO_BODY_COUNT - 1; ++body_idx) {
            float x = rand() % MAP_WIDTH / 4 + MAP_WIDTH / 8 + MAP_WIDTH / 2;
            float y = rand() % MAP_HEIGHT / 4 + MAP_HEIGHT / 8 + MAP_HEIGHT / 2;
            float dist = sqrt(pow((float) (x - (MAP_WIDTH / 4 + MAP_WIDTH / 2)), (float) 2) +
                              pow((float) (y - (MAP_HEIGHT / 4 + MAP_WIDTH / 2)), (float) 2));
            if (dist > MAP_WIDTH / 8) {
                --body_idx;
                continue;
            }

            bodies[1][body_idx].pos.x = x;
            bodies[1][body_idx].pos.y = y;
            bodies[1][body_idx].speed.x = 2;
            bodies[1][body_idx].speed.y = -1;
            bodies[1][global_idx + body_idx].mass = 0;
        }
        bodies[1][GALAXY_ONE_BODY_COUNT - 1].pos.x = MAP_WIDTH / 2;
        bodies[1][GALAXY_ONE_BODY_COUNT - 1].pos.y = MAP_HEIGHT / 2;
        bodies[1][GALAXY_ONE_BODY_COUNT - 1].mass = 100000000;

    }

    if (!galaxies) {

        unsigned int body_counts[] = {
                GALAXY_ONE_BODY_COUNT,
                GALAXY_TWO_BODY_COUNT
        };

        galaxies = galaxy_allocate(GALAXY_COUNT, &ocl, GALAXY_MAX_DEPTH, body_counts);
        galaxy_init(galaxies, &ocl, cells, GALAXY_MAX_DEPTH, GALAXY_ONE_THETA_VALUE, GALAXY_ONE_G_VALUE, bodies[0],
                    GALAXY_ONE_BODY_COUNT, MAP_WIDTH, MAP_HEIGHT, 0);
        galaxy_set_colors(galaxies, 0, 0.3, 0.3, QUADRANT, 0);
        galaxy_set_colors(galaxies, 0.6, 0.6, 1, BODY, 0);
        galaxy_init(galaxies, &ocl, cells + 1, GALAXY_MAX_DEPTH, GALAXY_TWO_THETA_VALUE, GALAXY_TWO_G_VALUE, bodies[1],
                    GALAXY_TWO_BODY_COUNT, MAP_WIDTH, MAP_HEIGHT, 1);
        galaxy_set_colors(galaxies, 0.5, 0.1, 0.1, QUADRANT, 1);
        galaxy_set_colors(galaxies, 0.8, 0.2, 0.2, BODY, 1);
        return;
    }

    struct timespec start_spec;
    clock_gettime(CLOCK_MONOTONIC, &start_spec);

    if (!KEYS.paused) {
        galaxy_resolve(galaxies, &ocl);
        galaxy_compute(galaxies, &ocl);
    }

    glClear(GL_COLOR_BUFFER_BIT);

    for (size_t idx = 0; idx < GALAXY_COUNT; ++idx) {

        galaxy_recover_bodies(galaxies, &ocl, bodies[idx], idx);
        galaxy_recover_cells(galaxies, &ocl, cells[idx], idx);

        glLineWidth(1);
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

        if (KEYS.grid) {
            glColor3f(galaxies->quadrant_color[idx].r,
                      galaxies->quadrant_color[idx].g,
                      galaxies->quadrant_color[idx].b);
            ogl_draw_quadrants(cells[idx], galaxies->cell_count[idx]);
        }

        glBegin(GL_POINTS);
        glLineWidth(1);
        glColor3f(galaxies->body_color[idx].r,
                  galaxies->body_color[idx].g,
                  galaxies->body_color[idx].b);

        ogl_draw_bodies(bodies[idx], galaxies->body_count[idx]);

        glEnd();

    }

    if (KEYS.grid) {
        glColor3f(1, 1, 1);
        glRasterPos2i(5, 5);

        char fps[50];

        snprintf(fps, 50, "%g", KEYS.fps);

        for (size_t idx = 0; fps[idx] != 0; ++idx) {
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_10, fps[idx]);
        }
    }

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