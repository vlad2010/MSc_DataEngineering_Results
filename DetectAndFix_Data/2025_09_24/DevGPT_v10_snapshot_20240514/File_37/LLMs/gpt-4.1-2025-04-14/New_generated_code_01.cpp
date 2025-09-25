printf("\x1b[16;%iH%s\n", (20 - ALGO_TEXT[1].length()/2), ALGO_TEXT[1].c_str());
  printf("\x1b[19;1H%s\n", DESCRIPTION_TEXT[0].c_str());