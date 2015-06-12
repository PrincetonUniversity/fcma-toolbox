/*
 This file is part of the Princeton FCMA Toolbox
 Copyright (c) 2013 the authors (see AUTHORS file)
 For license terms, please see the LICENSE file.
 */
#include <iostream>
#include <sstream>
#include <mpi.h>
#include "ErrorHandling.h"

void ContinueWithError(const std::ostringstream& os, ErrorType error /* = ERROR_NOEXIT */)
{
    std::cerr<<"Nonfatal errorcode "<<error<<": "<<os.str()<<std::endl;
    std::cerr<<"continuing..."<<std::endl;
}

void ExitWithError(const std::ostringstream& os, ErrorType error /*= ERROR_FATAL */)
{
    std::cerr<<"Fatal errorcode "<<error<<": "<<os.str()<<std::endl;
    std::cerr<<"exiting..."<<std::endl;
    int finalized;
    MPI_Finalized(&finalized);
    if (!finalized)
    {
        int initialized;
        MPI_Initialized(&initialized);
        if (initialized)
        {
            MPI_Abort(MPI_COMM_WORLD, error);
        }
    }
    exit(error);
}
