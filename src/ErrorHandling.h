/*
 This file is part of the Princeton FCMA Toolbox
 Copyright (c) 2013 the authors (see AUTHORS file)
 For license terms, please see the LICENSE file.
 */


#ifndef __ErrorHandling__
#define __ErrorHandling__

#define FATAL(x) do { std::ostringstream os; os << x; ExitWithError(os); } while (false)
#define NONFATAL(x) do { std::ostringstream os; os << x; ContinueWithError(os); } while (false)

enum ErrorType {
    ERROR_NONE,
    ERROR_NOEXIT,
    ERROR_FATAL,
    ERROR_FATAL_PARAM,
    NUM_ERRORS
};

void ContinueWithError(const std::ostringstream& os, ErrorType error = ERROR_NOEXIT);
void ExitWithError(const std::ostringstream& os, ErrorType error = ERROR_FATAL) __attribute__((__noreturn__));

#endif /* defined(__ErrorHandling__) */
