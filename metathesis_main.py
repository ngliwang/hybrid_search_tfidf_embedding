def find_anagrams(word_list):
    anagra_list = []
    anagram_dict = {}
    for word in list(set(word_list)):
        clean_sort = ''.join(sorted(word.lower()))

        # if clean version is not in the dictionary, add it
        if clean_sort not in anagram_dict.keys():
            anagram_dict[clean_sort] = [word]
        
        # if clean version is already in the dictionary, append the word to the list
        else:
            anagram_dict[clean_sort].append(word)

    # for each value in the dictionary, if the length of the value is greater than 1, append it to the list 
    for value in anagram_dict.values():
        
        # q: why must be bigger than 1? why can't it be >2?
        # a: because if it's >2, then it will only return the anagrams that have 3 or more words

        if len(value) > 1:
            anagra_list.append(list(value))
    
    return anagra_list

def find_anagrams2(word_list):
    anagra_list = []
    anagram_dict = {}
    for word in list(set(word_list)):
        clean_sort = ''.join(sorted(word.lower()))

        # if clean version is not in the dictionary, add it
        if clean_sort not in anagram_dict.keys():
            anagram_dict[clean_sort] = [word]
        
        # if clean version is already in the dictionary, append the word to the list
        else:
            anagram_dict[clean_sort].append(word)

    # for each value in the dictionary, if the length of the value is greater than 1, append it to the list 
    for value in anagram_dict.values():
        
        # q: why must be bigger than 1? why can't it be >2?
        # a: because if it's >2, then it will only return the anagrams that have 3 or more words

        if len(value) > 1:
            anagra_list.append(list(value))
    for alist in anagra_list:
        elementcount = len(alist)
        for i in range(elementcount):
            baseword=alist[i]
            for j in range(i+1,elementcount):
                comparedword=alist[j]
                baseworddict={}
                comparedworddict={}
                h=0
                for char in baseword:
                    baseworddict[h]=char
                    h+=1
                k=0
                for char in comparedword:
                    comparedworddict[k]=char
                    k+=1
                indicator=0
                for key in baseworddict.keys():
                    if baseworddict[key]==comparedworddict[key]:
                        continue
                    else: 
                        indicator+=1
                if indicator==2:
                    print(baseword,comparedword)
    return anagra_list

def main():
    word_list = ['cat', 'dog', 'tac', 'god', 'act', 'gdo', 'xyz', 'yxz', 'uured']
    print(find_anagrams2(word_list))


if __name__ == '__main__':
    main()