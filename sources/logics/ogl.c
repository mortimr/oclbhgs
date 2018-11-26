//
// Created by mortimr on 25/11/18.
//


#include "../headers/ogl.h"

ogl OGL;

void idle() {}

void keys_cb(unsigned char key, int x, int y) {

    (void) x;
    (void) y;

    switch (key) {
        case ' ':
            OGL.keys->paused = !OGL.keys->paused;
            break;
        case 'g':
            OGL.keys->grid = !OGL.keys->grid;
            break;
        case 'w':
            OGL.keys->fps = OGL.keys->fps < 480.0f ? OGL.keys->fps * 2.0f : 480.0f;
            break;
        case 's':
            OGL.keys->fps = OGL.keys->fps / 2.0f < 15.0f ? 15.0f : OGL.keys->fps / 2.0f;
            break;
        case 27:
            glutDestroyWindow(OGL.window);
            exit(0);
    }
}


void ogl_draw_bodies(body *bodies, size_t body_count) {

    for (size_t idx = 0; idx < body_count; ++idx) {

        if ((bodies[idx].pos.x > OGL.sim_x || bodies[idx].pos.y > OGL.sim_y) ||
            (bodies[idx].pos.x < 0 || bodies[idx].pos.y < 0)) {
            continue;
        };

        glVertex2f(((bodies[idx].pos.x / OGL.ratio_x)), ((bodies[idx].pos.y / OGL.ratio_y)));

    }

}

void ogl_draw_quadrants(cell *cells, size_t cell_count) {

    for (size_t idx = 0; idx < cell_count; ++idx) {

        if (cells[idx].active) {

            glRectf((cells[idx].pos.x / OGL.ratio_x),
                    (cells[idx].pos.y / OGL.ratio_y),
                    ((cells[idx].pos.x + cells[idx].size.x) / OGL.ratio_x),
                    ((cells[idx].pos.y + cells[idx].size.y) / OGL.ratio_y));

        }

    }

}

void ogl_init(void (*display)(),
              int *argc,
              char **argv,
              float sim_x,
              float sim_y,
              keys *keys_data) {

    glutInit((int *) argc, (char **) argv);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_MULTISAMPLE);
    glutInitWindowPosition(100, 100);
    glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
    OGL.window = glutCreateWindow("OpenCL Barnes Hut Galaxy Simulator");
    glutIdleFunc(&idle);
    glClearColor(0.0, 0.0, 0.0, 0.0);
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0.0, (double) WINDOW_WIDTH, 0.0, (double) WINDOW_HEIGHT);
    glutDisplayFunc(display);
    glutKeyboardFunc(keys_cb);
    OGL.sim_x = sim_x;
    OGL.sim_y = sim_y;
    OGL.ratio_x = sim_x / WINDOW_WIDTH;
    OGL.ratio_y = sim_y / WINDOW_HEIGHT;
    OGL.keys = keys_data;
    glutMainLoop();

}
