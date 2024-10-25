if (!File.Exists(setupFilePath))
{
    Console.WriteLine("Setup file not found. Please check the file path.");
    return;
}

if (!File.Exists(teamFilePath))
{
    Console.WriteLine("Team file not found. Please check the file path.");
    return;
}
