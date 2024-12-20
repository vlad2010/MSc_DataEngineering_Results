#define fr(n) for (int i = 0; i < n; i++)
#define fe(x,n) for (int i = x; i < n; i++)
#define freq(x,n) for (int i = x; i <= n; i++)
#define ll long long int
#define endl "\n"
#define pb push_back
#define read(x) int x; cin >> x
#define readl(x) ll x; cin >> x
#define readvi(v,n) vector<int> v;fr(n){read(x);v.pb(x);}
#define readvl(v,n) vector<ll> v;fr(n){readl(x);v.pb(x);}
#define p(ans) cout << ans << endl
#define yes p("YES")
#define no p("NO")
const ld PI = 3.14159265358979323846L;
const ld E = 2.71828182845904523536L;
const ll mod = 1000000007;

using namespace std;
//solved
void solve()
{
    read(n);
    read(m);
    read(k);
    read(h);
    vector<int> v(n);
    fr(n)
    {
        cin >> v[i];
        v[i] = abs(v[i] - h);
    }
    int ans = 0;
    fr(n) {
        if (v[i] == 0)
            continue;
        int x = v[i] % k;
        if (x == 0) {
            x = v[i] / k;
            if (x < m) {
                ans++;
            }
        }
    }
    p(ans);
}
