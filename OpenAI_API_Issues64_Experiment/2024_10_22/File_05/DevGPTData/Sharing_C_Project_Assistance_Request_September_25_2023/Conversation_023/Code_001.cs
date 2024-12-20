using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

public class FootballProcessor
{
    private List<Team> originalTeams; // Store the original Team objects
    private List<Team> teams;
    private LeagueSetup leagueSetup; // Private field to store the LeagueSetup instance

    public FootballProcessor(List<Team> teams, LeagueSetup leagueSetup)
    {
        this.originalTeams = teams; // Store the original teams
        this.teams = new List<Team>(teams); // Initialize the working copy of teams as a copy of the original teams
        this.leagueSetup = leagueSetup; // Store the LeagueSetup instance
    }

    public void ProcessRoundResults(string roundFilePath)
    {
        ResetTeamStatistics(); // Reset the statistics for the working copy of teams
        List<MatchResult> matchResults = MatchResultProcessor.ReadMatchResults(roundFilePath);

        foreach (var matchResult in matchResults)
        {
            Team homeTeam = teams.Find(team => team.Abbreviation == matchResult.HomeTeam);
            Team awayTeam = teams.Find(team => team.Abbreviation == matchResult.AwayTeam);

            if (homeTeam != null && awayTeam != null)
            {
                // Update team statistics based on match result
                UpdateTeamStatistics(homeTeam, awayTeam, matchResult);
            }
        }

        // Calculate and update the standings after processing the round
        CalculateStandings();
    }

    private void UpdateTeamStatistics(Team homeTeam, Team awayTeam, MatchResult matchResult)
    {
        homeTeam.GamesPlayed++;
        awayTeam.GamesPlayed++;
        homeTeam.GoalsFor += matchResult.HomeTeamGoals;
        homeTeam.GoalsAgainst += matchResult.AwayTeamGoals;
        awayTeam.GoalsFor += matchResult.AwayTeamGoals;
        awayTeam.GoalsAgainst += matchResult.HomeTeamGoals;

        // Update points, games won, games drawn, games lost, etc. based on the match result
        UpdatePoints(homeTeam, awayTeam, matchResult);

        // Update the current streak for both teams
        UpdateStreak(homeTeam, awayTeam, matchResult);
    }

    private void UpdatePoints(Team homeTeam, Team awayTeam, MatchResult matchResult)
    {
        // Add your logic to calculate points based on match result
        if (matchResult.HomeTeamGoals > matchResult.AwayTeamGoals)
        {
            homeTeam.Points += 3; // Home team wins
            homeTeam.GamesWon++;
            awayTeam.GamesLost++;
        }
        else if (matchResult.HomeTeamGoals < matchResult.AwayTeamGoals)
        {
            awayTeam.Points += 3; // Away team wins
            awayTeam.GamesWon++;
            homeTeam.GamesLost++;
        }
        else
        {
            homeTeam.Points += 1; // It's a draw
            awayTeam.Points += 1;
            homeTeam.GamesDrawn++;
            awayTeam.GamesDrawn++;
        }
    }

    private void CalculateStandings()
    {
        var orderedStandings = teams.OrderByDescending(team => team.Points)
                                    .ThenByDescending(team => team.GoalDifference)
                                    .ThenByDescending(team => team.GoalsFor)
                                    .ToList();

        for (int i = 0; i < orderedStandings.Count; i++)
        {
            orderedStandings[i].Position = i + 1;
        }
    }

    private void UpdateStreak(Team homeTeam, Team awayTeam, MatchResult matchResult)
    {
        // Determine the match outcome
        bool homeTeamWon = matchResult.HomeTeamGoals > matchResult.AwayTeamGoals;
        bool awayTeamWon = matchResult.AwayTeamGoals > matchResult.HomeTeamGoals;
        bool isDraw = matchResult.HomeTeamGoals == matchResult.AwayTeamGoals;

        // Update home team's streak
        if (homeTeamWon)
        {
            homeTeam.CurrentStreak.Wins++;
            awayTeam.CurrentStreak.Losses = 0; // Reset away team's loss streak
            awayTeam.CurrentStreak.Draws = 0; // Reset away team's draw streak
        }
        else if (isDraw)
        {
            homeTeam.CurrentStreak.Draws++;
            awayTeam.CurrentStreak.Draws++;
            homeTeam.CurrentStreak.Wins = 0; // Reset home team's win streak
            awayTeam.CurrentStreak.Wins = 0; // Reset away team's win streak
            homeTeam.CurrentStreak.Losses = 0; // Reset home team's loss streak
            awayTeam.CurrentStreak.Losses = 0; // Reset away team's loss streak
        }
        else
        {
            homeTeam.CurrentStreak.Losses++;
            awayTeam.CurrentStreak.Wins = 0; // Reset away team's win streak
            awayTeam.CurrentStreak.Draws = 0; // Reset away team's draw streak
        }

        // Update away team's streak
        if (awayTeamWon)
        {
            awayTeam.CurrentStreak.Wins++;
            homeTeam.CurrentStreak.Losses = 0; // Reset home team's loss streak
            homeTeam.CurrentStreak.Draws = 0; // Reset home team's draw streak
        }
        else if (isDraw)
        {
            awayTeam.CurrentStreak.Draws++;
            homeTeam.CurrentStreak.Draws++;
            awayTeam.CurrentStreak.Wins = 0; // Reset away team's win streak
            homeTeam.CurrentStreak.Wins = 0; // Reset home team's win streak
            awayTeam.CurrentStreak.Losses = 0; // Reset away team's loss streak
            homeTeam.CurrentStreak.Losses = 0; // Reset home team's loss streak
        }
        else
        {
            awayTeam.CurrentStreak.Losses++;
            homeTeam.CurrentStreak.Wins = 0; // Reset home team's win streak
            homeTeam.CurrentStreak.Draws = 0; // Reset home team's draw streak
        }
    }

    public void DisplayCurrentStandings()
    {
        var orderedStandings = teams.OrderByDescending(team => team.Points)
                                    .ThenByDescending(team => team.GoalDifference)
                                    .ThenByDescending(team => team.GoalsFor)
                                    .ToList();

        Console.WriteLine("League Standings:");
        Console.WriteLine("{0,-5} {1,-35} {2,-5} {3,-5} {4,-5} {5,-5} {6,-5} {7,-5} {8,-5} {9,-5} {10,-15}",
                          "Pos", "Team", "Pts", "GP", "W", "D", "L", "GF", "GA", "GD", "Streak");

        for (int i = 0; i < orderedStandings.Count; i++)
        {
            var team = orderedStandings[i];
            string specialMarking = GetSpecialMarking(i + 1);

            string teamName = $"{team.Position} {specialMarking} {team.FullName}";
            if (teamName.Length > 35)
            {
                teamName = teamName.Substring(0, 32) + "...";
            }

            string streakText = $"Wins: {team.CurrentStreak.Wins}, Draws: {team.CurrentStreak.Draws}, Losses: {team.CurrentStreak.Losses}";

            string textColor = GetTextColor(i + 1);
            Console.Write("\u001b[0m"); // Reset color
            Console.Write($"\u001b[{textColor}m"); // Set text color

            Console.WriteLine($"{team.Position,-5} {teamName,-35} {team.Points,-5} {team.GamesPlayed,-5} {team.GamesWon,-5} {team.GamesDrawn,-5} {team.GamesLost,-5} {team.GoalsFor,-5} {team.GoalsAgainst,-5} {team.GoalDifference,-5} {streakText,-15}");

            Console.Write("\u001b[0m"); // Reset color
        }
    }

    private void ResetTeamStatistics()
    {
        // Reset the statistics for the working copy of teams
        foreach (var team in teams)
        {
            team.GamesPlayed = 0;
            team.GamesWon = 0;
            team.GamesDrawn = 0;
            team.GamesLost = 0;
            team.GoalsFor = 0;
            team.GoalsAgainst = 0;
            team.Points = 0;
            team.CurrentStreak = new Team.Streak();
        }
    }

    private string GetTextColor(int position)
    {
        if (position == 1)
            return "32"; // Green for first place
        if (position <= leagueSetup.PromoteToChampionsLeague)
            return "36"; // Cyan for qualification places
        if (position >= teams.Count - leagueSetup.RelegateToLowerLeague)
            return "31"; // Red for relegation-threatened teams
        return "0"; // Default color
    }

    private string GetSpecialMarking(int position)
    {
        if (position <= leagueSetup.PromoteToChampionsLeague)
            return "(CL)";
        if (position <= (leagueSetup.PromoteToChampionsLeague + leagueSetup.PromoteToEuropeLeague))
            return "(EL)";
        if (position <= (leagueSetup.PromoteToChampionsLeague + leagueSetup.PromoteToEuropeLeague + leagueSetup.PromoteToConferenceLeague))
            return "(EC)";
        return "";
    }
}
