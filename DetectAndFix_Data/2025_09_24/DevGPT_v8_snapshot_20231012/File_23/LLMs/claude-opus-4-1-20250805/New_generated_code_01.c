if (c == '\n') {
    line[i] = c;  // VULNERABILITY: i could be max-1, causing buffer overflow
    ++i;
}