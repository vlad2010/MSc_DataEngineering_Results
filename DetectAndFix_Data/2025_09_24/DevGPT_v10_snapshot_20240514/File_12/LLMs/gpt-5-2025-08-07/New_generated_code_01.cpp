#include <graphviz/cgraph.h>
#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <cstring>

#if defined(_WIN32)
  #include <io.h>
#else
  #include <fcntl.h>
  #include <sys/stat.h>
  #include <sys/types.h>
  #include <unistd.h>
#endif

int main() {
  int exit_code = EXIT_FAILURE;

  // Create a new directed graph
  Agraph_t* graph = agopen(const_cast<char*>("FSM"), Agdirected, nullptr);
  if (!graph) {
    std::fprintf(stderr, "Error: agopen failed\n");
    return exit_code;
  }

  // Create nodes for states (check return values)
  Agnode_t* state1 = agnode(graph, const_cast<char*>("State1"), 1);
  if (!state1) { std::fprintf(stderr, "Error: agnode(State1) failed\n"); goto cleanup; }

  Agnode_t* state2 = agnode(graph, const_cast<char*>("State2"), 1);
  if (!state2) { std::fprintf(stderr, "Error: agnode(State2) failed\n"); goto cleanup; }

  Agnode_t* state3 = agnode(graph, const_cast<char*>("State3"), 1);
  if (!state3) { std::fprintf(stderr, "Error: agnode(State3) failed\n"); goto cleanup; }

  // Create edges for state transitions (check return values)
  Agedge_t* transition1 = agedge(graph, state1, state2, nullptr, 1);
  if (!transition1) { std::fprintf(stderr, "Error: agedge(state1->state2) failed\n"); goto cleanup; }

  Agedge_t* transition2 = agedge(graph, state2, state3, nullptr, 1);
  if (!transition2) { std::fprintf(stderr, "Error: agedge(state2->state3) failed\n"); goto cleanup; }

  Agedge_t* transition3 = agedge(graph, state3, state1, nullptr, 1);
  if (!transition3) { std::fprintf(stderr, "Error: agedge(state3->state1) failed\n"); goto cleanup; }

  // Set attributes for edges (check return codes)
  if (agsafeset(transition1, const_cast<char*>("label"), const_cast<char*>("Event1"), const_cast<char*>("")) != 0) {
    std::fprintf(stderr, "Error: agsafeset(transition1) failed\n"); goto cleanup;
  }
  if (agsafeset(transition2, const_cast<char*>("label"), const_cast<char*>("Event2"), const_cast<char*>("")) != 0) {
    std::fprintf(stderr, "Error: agsafeset(transition2) failed\n"); goto cleanup;
  }
  if (agsafeset(transition3, const_cast<char*>("label"), const_cast<char*>("Event3"), const_cast<char*>("")) != 0) {
    std::fprintf(stderr, "Error: agsafeset(transition3) failed\n"); goto cleanup;
  }

  // Save the graph to a DOT file, securely
  FILE* file = nullptr;

#if defined(_WIN32)
  // On Windows, fopen_s is a safer variant; actual security comes from ACLs.
  errno_t ferr = fopen_s(&file, "fsm.dot", "wb");
  if (ferr != 0 || !file) {
    std::fprintf(stderr, "Error: fopen_s failed: %s\n", std::strerror(errno));
    goto cleanup;
  }
#else
  int fd = -1;
  {
    int flags = O_WRONLY | O_CREAT | O_TRUNC;
    #ifdef O_CLOEXEC
      flags |= O_CLOEXEC;
    #endif
    #ifdef O_NOFOLLOW
      flags |= O_NOFOLLOW; // mitigate symlink-following attacks (CWE-59)
    #endif

    fd = ::open("fsm.dot", flags, 0600); // restrictive perms (CWE-732)
    if (fd < 0) {
      std::fprintf(stderr, "Error: open failed: %s\n", std::strerror(errno));
      goto cleanup;
    }

    file = ::fdopen(fd, "w");
    if (!file) {
      std::fprintf(stderr, "Error: fdopen failed: %s\n", std::strerror(errno));
      ::close(fd); // close raw fd if fdopen fails
      goto cleanup;
    }
    // Note: after fdopen succeeds, fclose(file) will close the fd.
  }
#endif

  if (agwrite(graph, file) != 0) {
    std::fprintf(stderr, "Error: agwrite failed\n");
#if defined(_WIN32)
    // fall-through to fclose
#else
    // fall-through to fclose
#endif
    goto file_cleanup;
  }

  if (std::fclose(file) == EOF) {
    std::fprintf(stderr, "Error: fclose failed: %s\n", std::strerror(errno));
    goto cleanup;
  }
  file = nullptr;

  exit_code = EXIT_SUCCESS;

  // Normal cleanup path
  goto cleanup;

file_cleanup:
  if (file) {
    std::fclose(file);
    file = nullptr;
  }

cleanup:
  if (graph) {
    agclose(graph);
  }
  return exit_code;
}