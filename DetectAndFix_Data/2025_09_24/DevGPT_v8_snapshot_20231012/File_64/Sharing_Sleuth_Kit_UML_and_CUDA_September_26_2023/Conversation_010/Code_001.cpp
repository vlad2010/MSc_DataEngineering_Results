/*
* The Sleuth Kit
*
* Brian Carrier [carrier <at> sleuthkit [dot] org]
* Copyright (c) 2014 Brian Carrier.  All rights reserved
*
*
* This software is distributed under the Common Public License 1.0
*/

#include "tsk_hashdb_i.h"
#include "tsk_hash_info.h"

/**
* \\file binsrch_index.cpp
* Functions common to all text hash databases (i.e. NSRL, HashKeeper, EnCase, etc.).
* Examples include index management and index-based lookup.
*/

// A mapping of initial hash digits to offsets in the index file is used to
// set the initial bounds of the binary search of the index file that is done
// for lookups. The mapping is from the first three digits (three nibbles) of 
// the hash, so there are 2 ^ 12 or 4096 possible entries.
static const size_t IDX_IDX_ENTRY_COUNT = 4096;
static const size_t IDX_IDX_SIZE = IDX_IDX_ENTRY_COUNT * sizeof(uint64_t);
...
