#ifndef _TCDF_H
#define _TCDF_H

#include <math.h>
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>


float tcdf(float, unsigned int);
float betai(float a, float b, float x);
float gammln(float xx);
float betacf(float a, float b, float x);
void nrerror(const char* error_text);

#endif
