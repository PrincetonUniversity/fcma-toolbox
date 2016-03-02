//
//  pni_error.c
//  pni_correlation
//
//  Created by Ben Singer on 1/19/16.
//  Copyright © 2016 Ben Singer. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include "pni_error.h"

static char msg[256];

static void verror(const char *fmt, va_list argp)
{
    fprintf(stderr, "error: ");
    vsprintf(msg, fmt, argp);
    perror(msg);
    fprintf(stderr, "\n");
}

void pni_error(const char *fmt, ...)
{
    va_list argp;
    va_start(argp, fmt);
    verror(fmt, argp);
    va_end(argp);
}

void pni_faterror(const char *fmt, ...)
{
    va_list argp;
    va_start(argp, fmt);
    verror(fmt, argp);
    va_end(argp);
    exit(EXIT_FAILURE);
}
