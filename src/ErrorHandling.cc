/*
 This file is part of the Princeton FCMA Toolbox
 Copyright (c) 2013 the authors (see AUTHORS file)
 For license terms, please see the LICENSE file.
 */
#include <iostream>
#include <sstream>
#include <mpi.h>
#include <unistd.h>
#include "ErrorHandling.h"

void ContinueWithError(const std::ostringstream& os,
                       ErrorType error /* = ERROR_NOEXIT */) {
  std::cerr << "Nonfatal errorcode " << error << ": " << os.str() << std::endl;
  std::cerr << "continuing..." << std::endl;
}

void ExitWithError(const std::ostringstream& os,
                   ErrorType error /*= ERROR_FATAL */) {
  std::cerr << "Fatal errorcode " << error << ": " << os.str() << std::endl;
  std::cerr << "exiting..." << std::endl;
  int finalized;
  MPI_Finalized(&finalized);
  if (!finalized) {
    int initialized;
    MPI_Initialized(&initialized);
    if (initialized) {
      MPI_Abort(MPI_COMM_WORLD, error);
    }
  }
  exit(error);
}

/* WaitForDebugAttach, from:
 *  <https://www.open-mpi.org/faq/?category=debugging>
 *
 *  This code will output a line to stdout outputting
 *  the name of the host where the process is running
 *  and the PID to attach to. It will then spin on the
 *  sleep() function forever waiting for you to attach
 *  with a debugger. Using sleep() as the inside of the
 *  loop means that the processor won't be pegged at
 *  100% while waiting for you to attach.
 *
 *  Once you attach with a debugger, you can set break
 *  points, set the loop variable to a nonzero value
 *  so it stops infinite loop and then continue.
 *
 *  If the host is $HOST and pid is $PID then do:
 *
 *  (head) ssh $HOST
 *  ($HOST) gdb --pid=$PID
 *  (gdb) break ErrorHandling.cpp:75  # line with sleep() below
 *  (gdb) set var i = 1
 *  (gdb) continue
 *  (gdb) step    #until you get up to interest (gdb always steps "in")
 *  (gdb) info locals   # see all local vars etc
 *
 */

void WaitForDebugAttach(void) {
  int i = 0;
  int namelen;
  char processor_name[MPI_MAX_PROCESSOR_NAME];

  MPI_Get_processor_name(processor_name, &namelen);
  printf("PID %d on %s ready for attach\n", getpid(), processor_name);
  fflush(stdout);
  while (0 == i) {
    sleep(5);
  }
}
