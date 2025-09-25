public static void Generate10(string filePath)
{
    List<string> upper = new List<string>
    {
        "FCK", "RFC", "FCM", "HBK", "AGF", "HOB"
    };
    
    List<string> lower = new List<string>
    {
        "AAB", "SIF", "VFF", "EFC", "BIF", "FCN"
    };

    int fileCount = 23;
    const int lowerGoals = 0;
    const int upperGoals = 6;
    
    int maxRounds = Math.Min(upper.Count, lower.Count) / 2;
    
    for (int i = 1; i < maxRounds + 1; i++)
    {
        int halfUpperCount = upper.Count / 2;
        int halfLowerCount = lower.Count / 2;
        
        // Creating first file
        StringBuilder csvContent1 = new StringBuilder();
        csvContent1.AppendLine("home,home goals,away,away goals");
        for (int j = 0; j < halfUpperCount; j++)
        {
            csvContent1.AppendLine(
                $"{upper[j % upper.Count]},{Rnd.Next(lowerGoals, upperGoals)},{upper[(j + i) % upper.Count]},{Rnd.Next(lowerGoals, upperGoals)}");
        }
        for (int j = 0; j < halfLowerCount; j++)
        {
            csvContent1.AppendLine(
                $"{lower[j % lower.Count]},{Rnd.Next(lowerGoals, upperGoals)},{lower[(j + i) % lower.Count]},{Rnd.Next(lowerGoals, upperGoals)}");
        }
        
        string fileName1 = $"round-{fileCount}.csv";
        string fullPath1 = Path.Combine(_rootDir, filePath, fileName1);
        File.WriteAllText(fullPath1, csvContent1.ToString());
        fileCount++;
        
        // Creating second file
        StringBuilder csvContent2 = new StringBuilder();
        csvContent2.AppendLine("home,home goals,away,away goals");
        for (int j = halfUpperCount; j < upper.Count; j++)
        {
            csvContent2.AppendLine(
                $"{upper[j % upper.Count]},{Rnd.Next(lowerGoals, upperGoals)},{upper[(j + i) % upper.Count]},{Rnd.Next(lowerGoals, upperGoals)}");
        }
        for (int j = halfLowerCount; j < lower.Count; j++)
        {
            csvContent2.AppendLine(
                $"{lower[j % lower.Count]},{Rnd.Next(lowerGoals, upperGoals)},{lower[(j + i) % lower.Count]},{Rnd.Next(lowerGoals, upperGoals)}");
        }

        string fileName2 = $"round-{fileCount}.csv";
        string fullPath2 = Path.Combine(_rootDir, filePath, fileName2);
        File.WriteAllText(fullPath2, csvContent2.ToString());
        fileCount++;
    }
}
