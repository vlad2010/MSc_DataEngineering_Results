   std::vector<std::string> binTree::zigzag(binTreeNode* r, bool childType, std::vector<std::string>& path)
   {
       // ...
       if (childType)
       {
           // left child
           vector1.push_back(r->location);
           leftPath = zigzag(r->left, true, vector1);
           rightPath = zigzag(r->right, false, path);
       }
       else
       {
           // right child
           vector1.push_back(r->location);
           leftPath = zigzag(r->left, true, path);
           rightPath = zigzag(r->right, false, vector1);
       }
       // ...
   }