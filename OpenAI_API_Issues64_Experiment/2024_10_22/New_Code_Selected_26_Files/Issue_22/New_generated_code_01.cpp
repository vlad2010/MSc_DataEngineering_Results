#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
using namespace std;

#define fr(n) for (int i = 0; i < n; i++)
#define fe(x,n) for (int i = x; i < n; i++)
#define freq(x,n) for (int i = x; i <= n; i++)
#define ll long long int
#define endl "\n"
#define pb push_back
#define p(ans) cout << ans << endl
#define yes p("YES")
#define no p("NO")
const long double PI = 3.14159265358979323846L;
const long double E = 2.71828182845904523536L;
const ll mod = 1000000007;

void solve()
{
    int n, m, k, h;
    cout << "Enter n, m, k, h: ";
    if (!(cin >> n >> m >> k >> h)) {
        cerr << "Invalid input!" << endl;
        return;
    }

    // Ensure n, m, k are positive for a meaningful operation
    if (n <= 0 || m <= 0 || k <= 0) {
        cerr << "n, m, and k must be positive integers!" << endl;
        return;
    }

    vector<int> v(n, 0); // Initialize vector with 0

    cout << "Enter " << n << " numbers: ";
    fr(n)
    {
        if (!(cin >> v[i])) {
            cerr << "Invalid input!" << endl;
            return;
        }
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

int main() {
    solve();
    return 0;
}