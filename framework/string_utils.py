def longest_common_substring(s1: str, s2: str) -> str:
    """
    Finds the longest common substring between two strings using dynamic programming.

    Args:
        s1: The first string.
        s2: The second string.

    Returns:
        The longest common substring. If multiple substrings have the same maximum length,
        any one of them can be returned. Returns an empty string if no common substring exists.
    """
    m = len(s1)
    n = len(s2)
    # Create a table to store lengths of longest common suffixes of substrings
    # dp[i][j] contains the length of the longest common suffix of s1[0...i-1] and s2[0...j-1]
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # To store the length of the longest common substring
    max_len = 0
    # To store the ending index of the longest common substring in s1
    end_index = 0

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > max_len:
                    max_len = dp[i][j]
                    end_index = i
            else:
                dp[i][j] = 0

    if max_len == 0:
        return ""
    else:
        return s1[end_index - max_len:end_index]
