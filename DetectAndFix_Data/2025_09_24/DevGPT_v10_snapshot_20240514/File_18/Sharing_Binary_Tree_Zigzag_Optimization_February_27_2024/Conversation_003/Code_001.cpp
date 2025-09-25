std::vector<std::string> binTree::zigzagLeft(binTreeNode* r, std::vector<std::string>& path)
{
    // Implement left zigzag traversal
}

std::vector<std::string> binTree::zigzagRight(binTreeNode* r, std::vector<std::string>& path)
{
    // Implement right zigzag traversal
}

std::vector<std::string> binTree::zigzag()
{
    std::vector<std::string> vector1; 
    std::vector<std::string> leftZag;
    std::vector<std::string> rightZag;

    leftZag = zigzagLeft(root, leftZag);
    rightZag = zigzagRight(root, rightZag);

    if (leftZag.size() + rightZag.size() == 0)
        return vector1;
    if (leftZag.size() > rightZag.size())
        return leftZag;
    return rightZag;
}