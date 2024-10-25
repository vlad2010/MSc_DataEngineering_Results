public class FootballProcessor
{
    private List<Team> originalTeams; // Store the original Team objects
    private List<Team> teams;
    private LeagueSetup leagueSetup;

    public FootballProcessor(List<Team> teams, LeagueSetup leagueSetup)
    {
        this.originalTeams = new List<Team>(teams); // Make a copy of the original teams
        this.teams = new List<Team>(teams); // Initialize the working copy of teams
        this.leagueSetup = leagueSetup;
    }

    // ...
}
