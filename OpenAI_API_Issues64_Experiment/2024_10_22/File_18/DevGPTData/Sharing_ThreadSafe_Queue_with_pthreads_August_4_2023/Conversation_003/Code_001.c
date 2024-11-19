/*
 * =====================================================================================
 *
 *       Filename:  prot_queue.h
 *
 *    Description:  This header file provides a thread-safe queue implementation for
 *                  generic data elements. It uses POSIX threads (pthreads) to ensure
 *                  thread safety. The queue allows for pushing and popping elements,
 *                  with the ability to block or non-block on pop operations.
 *                  Users are responsible for providing memory for the queue buffer
 *                  and ensuring its correct lifespan.
 *
 *        Version:  1.0
 *        Created:  [DATE]
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Your Name (or your organization), [EMAIL]
 *
 * =====================================================================================
 */
