#ifndef STRING_H
#define STRING_H

#include "pch.h"

namespace fe {
    /// <summary>
    /// Replaces every instance of the specified token/s in the target string.
    /// </summary>
    /// <param name="stringToSearch">String to search.</param>
    /// <param name="firstToReplace">First string to replace.</param>
    /// <param name="firstReplacement">First string to get replaced.</param>
    /// <param name="...otherPairsOfStringsToReplace">Other pairs of strings to replace.</param>
    /// <returns>String with replaced arguments.</returns>
    template <typename StringType, typename... OtherReplacements>
    static std::string Replace(StringType stringToSearch, std::string_view firstToReplace, std::string_view firstReplacement,
        OtherReplacements&&... otherPairsOfStringsToReplace)
    {
        static_assert ((sizeof... (otherPairsOfStringsToReplace) & 1u) == 0,
            "this function expects a list of pairs of strings as its arguments!");

        if constexpr (std::is_same<const StringType, const std::string_view>::value || std::is_same<const StringType, const char* const>::value)
        {
            return Replace(std::string(stringToSearch), firstToReplace, firstReplacement,
                std::forward<OtherReplacements>(otherPairsOfStringsToReplace)...);
        }
        else if constexpr (sizeof... (otherPairsOfStringsToReplace) == 0)
        {
            size_t pos = 0;

            for (;;)
            {
                pos = stringToSearch.find(firstToReplace, pos);

                if (pos == std::string::npos) {
                    return stringToSearch;
                }

                stringToSearch.replace(pos, firstToReplace.length(), firstReplacement);
                pos += firstReplacement.length();
            }
        }
        else
        {
            return Replace(Replace(std::move(stringToSearch), firstToReplace, firstReplacement),
                std::forward<OtherReplacements>(otherPairsOfStringsToReplace)...);
        }
    }

    /// <summary>
    /// Splits a string into the given array at a give token.
    /// </summary>
    /// <param name="string">Source string.</param>
    /// <param name="token">Token to split the string at.</param>
    /// <param name="out">Vector containing the split string's excluding the token.</param>
    static void Split(const std::string& string,
        std::string token,
        std::vector<std::string>& out)
    {
        out.clear();

        std::string temp;

        for (int i = 0; i < int(string.size()); i++)
        {
            std::string test = string.substr(i, token.size());

            if (test == token)
            {
                if (!temp.empty())
                {
                    out.push_back(temp);
                    temp.clear();
                    i += (int)token.size() - 1;
                }
                else
                {
                    out.push_back("");
                }
            }
            else if (i + token.size() >= string.size())
            {
                temp += string.substr(i, token.size());
                out.push_back(temp);
                break;
            }
            else
            {
                temp += string[i];
            }
        }
    }

    /// <summary>
    /// Converts all characters in the specified string to lowercase.
    /// </summary>
    /// <param name="string">String to format.</param>
    /// <returns>Formatted string.</returns>
    static std::string& ToLower(std::string& string)
    {
        std::transform(string.begin(), string.end(), string.begin(),
            [](const unsigned char c) { return std::tolower(c); });
        return string;
    }

    /// <summary>
    /// Checks if the specified string contains the specified token.
    /// </summary>
    /// <param name="string">String to check.</param>
    /// <param name="token">Token to look for.</param>
    /// <returns>True if the string contains the token, otherwise returns false.</returns>
    static bool Contains(std::string_view string, std::string_view token) {
        return string.find(token) != std::string::npos;
    }
}

#endif // !STRING_H