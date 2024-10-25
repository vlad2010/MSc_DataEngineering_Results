public class Team
{
    // Other properties...

    public List<Streak> Streaks { get; set; } = new List<Streak>(); // Store historical streaks

    public class Streak
    {
        public int Wins { get; set; }
        public int Draws { get; set; }
        public int Losses { get; set; }
    }
}
