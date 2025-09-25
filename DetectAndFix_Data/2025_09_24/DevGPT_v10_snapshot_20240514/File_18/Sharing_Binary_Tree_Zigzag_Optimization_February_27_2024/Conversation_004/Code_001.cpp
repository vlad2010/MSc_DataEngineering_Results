std::vector<std::string> binTree::zigzagLeft(binTreeNode* r, std::vector<std::string>& path)
{
    if (r == nullptr)
        return std::vector<std::string>();

    path.push_back(r->location);

    std::vector<std::string> leftPath = zigzagLeft(r->left, path);
    std::vector<std::string> rightPath = zigzagLeft(r->right, std::vector<std::string>());

    return (leftPath.size() > rightPath.size()) ? leftPath : rightPath;
}

std::vector<std::string> binTree::zigzagRight(binTreeNode* r, std::vector<std::string>& path)
{
    if (r == nullptr)
        return std::vector<std::string>();

    path.push_back(r->location);

    std::vector<std::string> leftPath = zigzagRight(r->left, std::vector<std::string>());
    std::vector<std::string> rightPath = zigzagRight(r->right, path);

    return (leftPath.size() > rightPath.size()) ? leftPath : rightPath;
}

std::vector<std::string> binTree::zigzag()
{
    std::vector<std::string> vector1; 
    std::vector<std::string> leftZag = zigzagLeft(root, vector1);
    std::vector<std::string> rightZag = zigzagRight(root, vector1);

    if (leftZag.size() + rightZag.size() == 0)
        return vector1;
    return (leftZag.size() > rightZag.size()) ? leftZag : rightZag;
}