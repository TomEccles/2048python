// Python2048Extension.cpp : Defines the exported functions for the DLL application.
//

#include <iostream>
#include "stdafx.h"
#ifdef _DEBUG
#undef _DEBUG
#include <Python.h>
#define _DEBUG
#else
#include <Python.h>
#endif
#include "Board.h"
#include "Roller.h"
#include "Python2048Extension.h"

static PyObject * pythonListFromBoard(Board b)
{
    long board[4][4];
    PyObject *rows = PyList_New(4);
    for (int i = 0; i < 4; i++) {
        PyObject *row = PyList_New(4);
        for (int j = 0; j < 4; j++) {
            PyList_SetItem(row, j, PyLong_FromLong(b.elementAt(i, j)));
        }
        PyList_SetItem(rows, i, row);
    }
    return rows;
}


static Board boardFromPythonList(PyObject *list)
{
    long board[4][4];
    for (int i = 0; i < 4; i++) {
        PyObject *row = PyList_GetItem(list, i);
        for (int j = 0; j < 4; j++) {
            PyObject *val = PyList_GetItem(row, j);
            long v = PyLong_AsLong(val);
            board[i][j] = v;
        }
    }
    Board b = Board(board);
    return b;
}

static PyObject* rollout_from_move(PyObject* self, PyObject* list)
{
    int moves = Roller().rolloutFromMove(boardFromPythonList(list));
    return PyLong_FromLong(moves);
}

static PyObject* can_move_top_row(PyObject* self, PyObject* list)
{
    if (boardFromPythonList(list).topRowCanMoveRight()) Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

static PyObject* greedy_move(PyObject* self, PyObject* list)
{
    Board b = boardFromPythonList(list);
    Roller().greedyMove(b);
    return pythonListFromBoard(b);
}

static PyMethodDef funcs[] = {
    { "roller", (PyCFunction)rollout_from_move,
    METH_O, "" },
    { "can_move_top_row", (PyCFunction)can_move_top_row,
    METH_O, "" },
    { "greedy_move", (PyCFunction)greedy_move,
    METH_O, "" },
    { NULL, NULL, 0, NULL }
};

static struct PyModuleDef cModPyDem =
{
    PyModuleDef_HEAD_INIT,
    "roller", /* name of module */
    "",          /* module documentation, may be NULL */
    -1,          /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    funcs
};

PyMODINIT_FUNC PyInit_roller(void)
{
    srand(time(NULL));
    return PyModule_Create(&cModPyDem);
}