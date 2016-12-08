/*
 This file is part of the Princeton FCMA Toolbox
 Copyright (c) 2013 the authors (see AUTHORS file)
 For license terms, please see the LICENSE file.
*/

#include "common.h"

/* correlates all subjects in a directory, using N-1 combinations,
 * and writes out resulting average matrix. Done for a given trial. */
void WriteAverageCorrelations(int nSubs, RawMatrix** r_matrix,
			    const char* maskFile1, const char* maskFile2,
                            Trial trial, const char* output_file);
