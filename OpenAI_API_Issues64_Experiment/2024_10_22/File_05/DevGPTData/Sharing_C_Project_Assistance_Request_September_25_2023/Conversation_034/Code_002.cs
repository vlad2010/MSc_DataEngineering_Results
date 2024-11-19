string streak = fields[11];
if (!string.IsNullOrEmpty(streak) && streak.Length >= 2)
{
    char streakResult = streak[0];
    int streakValue = int.Parse(streak.Substring(1));

    if (streakResult == 'W')
    {
        team.CurrentStreak.Wins = streakValue;
    }
    else if (streakResult == 'D')
    {
        team.CurrentStreak.Draws = streakValue;
    }
    else if (streakResult == 'L')
    {
        team.CurrentStreak.Losses = streakValue;
    }
}
