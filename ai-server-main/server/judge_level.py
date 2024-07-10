diseaseList = ['blood','bilirubin','urobilinogen','ketones','protein', 'nitrite', 'glucose','ph','sg','leukocytes']

mapping = {
    '다낭성신장염,방광염': ['blood', 'nitrite'], 
    '방광염,신우신염': ['blood', 'protein'], 
    '췌장염,갑상선항진증': ['bilirubin', 'protein'], 
    '신장질환,신우신염': ['protein'], 
    '만성신장질환,방광염': ['blood', 'protein'], 
    '건강검진': [],
    '당뇨': ['glucose', 'nitrite'], 
    '신장염,신우신염': ['protein'], 
    '빈혈,방광염': ['blood', 'protein'], 
    '다낭성신장염,신우신염': ['blood', 'protein'], 
    '당뇨,신장염': ['glucose', 'nitrite'], 
    '신장염,방광염': ['protein'], 
    '빈혈,신장염': ['blood', 'protein'], 
    '만성신장질환': ['blood', 'protein'], 
    '췌장염': ['bilirubin', 'protein'], 
    '만성췌장염,방광염': ['blood', 'protein'], 
    # '만성구토': ['blood', 'protein', 'nitrite', 'glucose'], 
    # '다낭성신장염': ['blood', 'protein'], 
    '만성췌장염,갑상선항진증': ['blood', 'protein'], 
    '신장염,빈혈': ['blood'], 
    '간염,췌장염': ['bilirubin'], 
    '급성간염,담도폐쇄': ['urobilinogen'], 
    '방광염,신장질환,전립선염': ['leukocytes'], 
    '신장질환,당뇨': ['ketones','protein', 'glucose'], 
    '신장염,사구체신염': ['blood', 'protein', 'nitrite'], 
    '방광염,요로결석': ['blood', 'protein', 'nitrite'], 
    '방광염,신장염,요로감염': ['blood', 'protein'], 
    '췌장염,간염,담도암': ['bilirubin','urobilinogen'], 
    '췌장염,간염,세포성황달': ['bilirubin','urobilinogen'], 
    '간염,황달,빈혈': ['bilirubin','urobilinogen'], 
    '췌장염,췌장암': ['bilirubin', 'glucose'], 
    '갑상선기능항진증': ['ketones', 'glucose'], 
    # '심부전': ['protein'], 
    # '뇌종양,뇌외상': ['blood', 'glucose'], 
    '간염,급만성간경변': ['blood', 'glucose']
 }

levelMapping = { 4 : '고위험', 3 : '위험', 2 : '주의', 1 : '관심', 0 : '', -1: '' }


def judgeLevel(disease, data):
    mainFactorIdx = [diseaseList.index(factor) for factor in mapping[disease]]
    factorValues = [data[factorIdx] for factorIdx in mainFactorIdx]
    largeFactorValue = 0 if factorValues == [] else max(factorValues)
    print(f"judgeLevel = {disease} : {mainFactorIdx}, {factorValues} --> {levelMapping[largeFactorValue]}")
    return levelMapping[largeFactorValue]