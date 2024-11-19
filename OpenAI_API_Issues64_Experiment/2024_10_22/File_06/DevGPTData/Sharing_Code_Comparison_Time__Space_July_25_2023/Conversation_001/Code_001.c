while (tt--) {
  int n, m, k, H;
  cin >> n >> m >> k >> H;
  int ans = 0;
  for (int i = 0; i < n; i++) {
    int h;
    cin >> h;
    if (h != H && h % k == H % k && abs(h - H) <= k * (m - 1)) {
      ans += 1;
    }
  }
  cout << ans << '\n';
}
