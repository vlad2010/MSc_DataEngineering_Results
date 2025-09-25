public void ProcessRoundResults(string roundFilePath)
{
    ResetTeamStatistics(); // Reset the statistics for the working copy of teams
    List<MatchResult> matchResults = MatchResultProcessor.ReadMatchResults(roundFilePath);

    // ...
}
