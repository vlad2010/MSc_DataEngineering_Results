public class Team
{
    public string Abbreviation { get; set; }
    public string FullName { get; set; }
    public string SpecialRanking { get; set; }
    public int Position { get; set; }
    public int GamesPlayed { get; set; }
    public int GamesWon { get; set; }
    public int GamesDrawn { get; set; }
    public int GamesLost { get; set; }
    public int GoalsFor { get; set; }
    public int GoalsAgainst { get; set; }
    public int GoalDifference => GoalsFor - GoalsAgainst;
    public int Points { get; set; }
    public Streak CurrentStreak { get; set; }

    public Team()
    {
        CurrentStreak = new Streak();
    }

    public Team(string abbreviation, string fullName, string specialRanking)
    {
        Abbreviation = abbreviation;
        FullName = fullName;
        SpecialRanking = specialRanking;
        Position = 0;
        GamesPlayed = 0;
        GamesWon = 0;
        GamesDrawn = 0;
        GamesLost = 0;
        GoalsFor = 0;
        GoalsAgainst = 0;
        Points = 0;
        CurrentStreak = new Streak();
    }

    public class Streak
    {
        public int Wins { get; set; }
        public int Draws { get; set; }
        public int Losses { get; set; }
    }

    public override string ToString()
    {
        return $"Wins: {CurrentStreak.Wins}, Draws: {CurrentStreak.Draws}, Losses: {CurrentStreak.Losses}";
    }
}
