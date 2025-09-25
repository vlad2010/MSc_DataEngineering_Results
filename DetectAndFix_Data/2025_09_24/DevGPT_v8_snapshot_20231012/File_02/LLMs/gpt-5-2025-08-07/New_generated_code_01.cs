using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Data.Entity;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Security.Cryptography;
using System.Threading.Tasks;
using System.Web.Http;
using System.Web.Http.Controllers;
using System.Web.Http.Description;
using System.Web.Http.Filters;

namespace SecureUsersApi
{
    // HTTPS enforcement (CWE-319)
    public class RequireHttpsAttribute : AuthorizationFilterAttribute
    {
        public override void OnAuthorization(HttpActionContext actionContext)
        {
            if (actionContext == null || actionContext.Request == null)
            {
                base.OnAuthorization(actionContext);
                return;
            }
            if (!string.Equals(actionContext.Request.RequestUri.Scheme, Uri.UriSchemeHttps, StringComparison.OrdinalIgnoreCase))
            {
                actionContext.Response = actionContext.Request.CreateResponse(
                    HttpStatusCode.Forbidden,
                    new { error = "HTTPS is required." }
                );
            }
        }
    }

    // Data models and EF context (example)
    public class User
    {
        public int Id { get; set; }

        [Required, StringLength(64)]
        public string UserName { get; set; }

        [Required, StringLength(256), EmailAddress]
        public string Email { get; set; }

        // Never return this to clients
        [Required, StringLength(1024)]
        public string PasswordHash { get; set; }

        public DateTimeOffset CreatedAt { get; set; } = DateTimeOffset.UtcNow;
    }

    public class AppDbContext : DbContext
    {
        public AppDbContext() : base("name=AppDbContext") { }
        public DbSet<User> Users { get; set; }
    }

    // DTOs (whitelist fields)
    public class UserReadDto
    {
        public int Id { get; set; }
        public string UserName { get; set; }
        public string Email { get; set; }
        public DateTimeOffset CreatedAt { get; set; }
    }

    public class CreateUserDto
    {
        [Required, StringLength(64), RegularExpression(@"^[A-Za-z0-9_\-\.]+$")]
        public string UserName { get; set; }

        [Required, StringLength(256), EmailAddress]
        public string Email { get; set; }

        // Only include if this endpoint sets passwords. Otherwise remove.
        [Required, StringLength(128, MinimumLength = 12)]
        public string Password { get; set; }
    }

    // Password hasher using PBKDF2 (CWE-256, CWE-916)
    public static class SecurePasswordHasher
    {
        // Adjust iteration count as per current guidance (e.g., >= 210,000 for PBKDF2-HMAC-SHA256).
        private const int Iterations = 210_000;
        private const int SaltSize = 16; // 128-bit
        private const int KeySize = 32;  // 256-bit

        public static string Hash(string password)
        {
            if (password == null) throw new ArgumentNullException(nameof(password));
            using (var rng = RandomNumberGenerator.Create())
            {
                var salt = new byte[SaltSize];
                rng.GetBytes(salt);
                using (var pbkdf2 = new Rfc2898DeriveBytes(password, salt, Iterations, HashAlgorithmName.SHA256))
                {
                    var key = pbkdf2.GetBytes(KeySize);
                    // Format: pbkdf2-sha256$ITERATIONS$BASE64(SALT)$BASE64(KEY)
                    return $"pbkdf2-sha256${Iterations}${Convert.ToBase64String(salt)}${Convert.ToBase64String(key)}";
                }
            }
        }

        public static bool Verify(string password, string hash)
        {
            if (string.IsNullOrWhiteSpace(hash) || password == null) return false;
            var parts = hash.Split('$');
            if (parts.Length != 4 || !parts[0].Equals("pbkdf2-sha256", StringComparison.OrdinalIgnoreCase)) return false;
            var iterations = int.Parse(parts[1]);
            var salt = Convert.FromBase64String(parts[2]);
            var key = Convert.FromBase64String(parts[3]);

            using (var pbkdf2 = new Rfc2898DeriveBytes(password, salt, iterations, HashAlgorithmName.SHA256))
            {
                var computed = pbkdf2.GetBytes(key.Length);
                return CryptographicOperations.FixedTimeEquals(computed, key);
            }
        }
    }

    [Authorize] // Require authentication for all actions (CWE-306)
    [RequireHttps] // Enforce TLS (CWE-319)
    [RoutePrefix("api/users")]
    public class UsersController : ApiController
    {
        private readonly AppDbContext _db = new AppDbContext();

        // GET api/users?page=1&pageSize=50
        // Authorization scoped as needed; listing users is typically privileged (CWE-862, CWE-200).
        [Authorize(Roles = "Admin,UserReader")]
        [HttpGet]
        [ResponseType(typeof(IEnumerable<UserReadDto>))]
        [Route("")]
        public async Task<IHttpActionResult> GetUsers([FromUri] int page = 1, [FromUri] int pageSize = 50)
        {
            try
            {
                // Bound and cap pagination to prevent resource exhaustion (CWE-770)
                if (page < 1) page = 1;
                pageSize = Math.Min(Math.Max(pageSize, 1), 100);

                var users = await _db.Users
                    .AsNoTracking()
                    .OrderBy(u => u.Id)
                    .Skip((page - 1) * pageSize)
                    .Take(pageSize)
                    .Select(u => new UserReadDto
                    {
                        Id = u.Id,
                        UserName = u.UserName,
                        Email = u.Email,
                        CreatedAt = u.CreatedAt
                    })
                    .ToListAsync();

                // Disable caching for sensitive data (CWE-524)
                var response = Request.CreateResponse(HttpStatusCode.OK, users);
                response.Headers.CacheControl = new CacheControlHeaderValue
                {
                    NoCache = true,
                    NoStore = true,
                    MustRevalidate = true
                };
                response.Headers.Pragma.ParseAdd("no-cache");
                response.Content.Headers.ContentType = new MediaTypeHeaderValue("application/json");
                return ResponseMessage(response);
            }
            catch
            {
                // Avoid leaking internals (CWE-209)
                return InternalServerError();
            }
        }

        // GET api/users/123
        [Authorize(Roles = "Admin,UserReader")]
        [HttpGet]
        [ResponseType(typeof(UserReadDto))]
        [Route("{id:int}", Name = "GetUserById")]
        public async Task<IHttpActionResult> GetUserById(int id)
        {
            try
            {
                var user = await _db.Users.AsNoTracking().Where(u => u.Id == id).Select(u => new UserReadDto
                {
                    Id = u.Id,
                    UserName = u.UserName,
                    Email = u.Email,
                    CreatedAt = u.CreatedAt
                }).FirstOrDefaultAsync();

                if (user == null) return NotFound();
                var response = Request.CreateResponse(HttpStatusCode.OK, user);
                response.Headers.CacheControl = new CacheControlHeaderValue
                {
                    NoCache = true,
                    NoStore = true,
                    MustRevalidate = true
                };
                response.Content.Headers.ContentType = new MediaTypeHeaderValue("application/json");
                return ResponseMessage(response);
            }
            catch
            {
                return InternalServerError();
            }
        }

        // POST api/users
        // If your API uses cookie-based auth, add CSRF protections (CWE-352), e.g., validate an anti-forgery token header.
        // If you use stateless Bearer/JWT and do not use cookies, CSRF risk is reduced (still avoid enabling CORS broadly).
        [Authorize(Roles = "Admin,UserCreator")]
        [HttpPost]
        [ResponseType(typeof(UserReadDto))]
        [Route("")]
        public async Task<IHttpActionResult> CreateUser([FromBody] CreateUserDto userDto)
        {
            try
            {
                if (userDto == null) return BadRequest("Request body is required.");
                if (!ModelState.IsValid) return BadRequest(ModelState); // (CWE-20)

                // Ensure uniqueness; return generic conflicts to avoid user enumeration (CWE-200)
                var exists = await _db.Users.AnyAsync(u =>
                    u.UserName == userDto.UserName || u.Email == userDto.Email);
                if (exists)
                {
                    return Content(HttpStatusCode.Conflict, new { error = "User with same username or email already exists." });
                }

                // Hash password before storing (CWE-256, CWE-916)
                var passwordHash = SecurePasswordHasher.Hash(userDto.Password);

                var user = new User
                {
                    UserName = userDto.UserName,
                    Email = userDto.Email,
                    PasswordHash = passwordHash
                };

                _db.Users.Add(user);
                await _db.SaveChangesAsync();

                var resultDto = new UserReadDto
                {
                    Id = user.Id,
                    UserName = user.UserName,
                    Email = user.Email,
                    CreatedAt = user.CreatedAt
                };

                // Return 201 with location header
                return CreatedAtRoute("GetUserById", new { id = user.Id }, resultDto);
            }
            catch
            {
                // Avoid leaking internals
                return InternalServerError();
            }
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing) _db.Dispose();
            base.Dispose(disposing);
        }
    }
}