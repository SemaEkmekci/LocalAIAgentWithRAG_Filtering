# main.py değişiklik
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import search_data, cities, tax_offices, companies, direct_search

model = OllamaLLM(model="llama3.2")

template = """
Sen Türkçe şehir, vergi daireleri ve firmalar hakkındaki sorguları anlama konusunda uzman bir asistansın.
Kullanıcının sorgusu: {query}

Bu sorguya göre kullanıcının hangi şehirler, vergi daireleri ve firmalardan bahsettiğini belirle.
ÖNEMLI: Kullanıcı sorgusunda açıkça belirtilmeyen hiçbir öğeyi dahil etme!
Örneğin, kullanıcı sadece "Konya'daki firmalar" derse, şehir olarak Konya belirtilmeli ama herhangi bir vergi dairesi veya firma belirtilmemelidir.
Veya "Meram Süt Ürünleri" derse, sadece bu firma belirtilmeli, kullanıcı açıkça sormadıysa onun şehri, vergi dairesi veya firmalar otomatik eklenmemelidir.

Cevabını şu formatta ver:
{{
  "cities": [sorguda açıkça belirtilen şehirlerin ID'leri veya isimleri],
  "tax": [sorguda açıkça belirtilen vergi dairelerinin ID'leri veya isimleri],
  "companies": [sorguda açıkça belirtilen firmaların ID'leri veya isimleri]
}}

Mevcut şehirler:
{city_list}

Mevcut vergi daireleri:
{tax_list}

Mevcut firmalar:
{company_list}

Hassas ol - yalnızca sorguda açıkça belirtilen öğeleri dahil et.
"""

# Şablon için şehir, vergi dairesi ve firma listelerini oluştur
city_list = "\n".join([f"ID: {c['id']}, İsim: {c['city']}" for c in cities])
tax_list = "\n".join([f"ID: {t['id']}, İsim: {t['name']}" for t in tax_offices])
company_list = "\n".join([f"ID: {c['id']}, İsim: {c['name']}" for c in companies])

# Şablon oluştur
prompt = ChatPromptTemplate.from_template(template)

# Zincir oluştur
chain = prompt | model

def enhanced_search(query):
    """
    Bu fonksiyon, arama doğruluğunu artırmak için birkaç arama yöntemini birleştirir
    ve sadece sorguda açıkça belirtilen öğeleri döndürür
    """
    # Önce doğrudan arama yap (regex ile)
    direct_results = direct_search(query)
    
    # Vektör araması yap
    vector_results = search_data(query)
    
    # LLM analizi yap - açıkça belirtilen öğeleri anlamak için
    llm_response = chain.invoke({
        "query": query,
        "city_list": city_list,
        "tax_list": tax_list,
        "company_list": company_list
    })
    
    # Tüm sonuçları birleştirmeden önce işle
    final_results = {
        "cities": [],
        "tax": [],
        "companies": []
    }
    
    try:
        # LLM yanıtını JSON olarak parse etmeye çalış
        import json
        import re
        
        # Yanıttaki JSON benzeri yapıyı bul
        json_match = re.search(r'{[\s\S]*}', llm_response)
        if json_match:
            llm_json = json.loads(json_match.group(0))
            
            # String değerlerini karşılık gelen ID'lere dönüştür
            city_ids = []
            for city_value in llm_json.get("cities", []):
                if isinstance(city_value, int):
                    # Eğer zaten bir ID ise, doğrudan kullan
                    city_ids.append(city_value)
                else:
                    # Eğer bir string (şehir adı) ise, karşılık gelen ID'yi bul
                    city_name = city_value.lower()
                    matching_city = next((c for c in cities if c["city"].lower() == city_name), None)
                    if matching_city:
                        city_ids.append(matching_city["id"])
            
            tax_ids = []
            for tax_value in llm_json.get("tax", []):
                if isinstance(tax_value, int):
                    # Eğer zaten bir ID ise, doğrudan kullan
                    tax_ids.append(tax_value)
                else:
                    # Eğer bir string (vergi dairesi adı) ise, karşılık gelen ID'yi bul
                    tax_name = tax_value.lower()
                    matching_tax = next((t for t in tax_offices if tax_name in t["name"].lower()), None)
                    if matching_tax:
                        tax_ids.append(matching_tax["id"])
            
            company_ids = []
            for company_value in llm_json.get("companies", []):
                if isinstance(company_value, int):
                    # Eğer zaten bir ID ise, doğrudan kullan
                    company_ids.append(company_value)
                else:
                    # Eğer bir string (firma adı) ise, karşılık gelen ID'yi bul
                    company_name = company_value.lower()
                    matching_company = next((c for c in companies if company_name in c["name"].lower()), None)
                    if matching_company:
                        company_ids.append(matching_company["id"])
            
            # LLM sonuçlarını final_results'a ekle
            final_results["cities"].extend(city_ids)
            final_results["tax"].extend(tax_ids)
            final_results["companies"].extend(company_ids)
    except Exception as e:
        print(f"LLM yanıtını işlerken hata: {e}")
    
    # Doğrudan arama sonuçlarını ekle
    for key in final_results:
        for item_id in direct_results.get(key, []):
            if item_id not in final_results[key]:
                final_results[key].append(item_id)
    
    # Vektör sonuçlarını ekle ama bunlar düşük öncelikli
    # LLM ve direct search kesin sonuç vermezse
    if not any(final_results.values()):
        for key in final_results:
            for item_id in vector_results.get(key, []):
                if item_id not in final_results[key]:
                    final_results[key].append(item_id)
    
    # Sonuçları doğrula - gerçekten veritabanında var mı?
    valid_results = {
        "cities": [],
        "tax": [],
        "companies": []
    }
    
    # Şehirleri doğrula
    for city_id in final_results["cities"]:
        if any(c["id"] == city_id for c in cities):
            valid_results["cities"].append(city_id)
    
    # Vergi dairelerini doğrula
    for tax_id in final_results["tax"]:
        if any(t["id"] == tax_id for t in tax_offices):
            valid_results["tax"].append(tax_id)
    
    # Firmaları doğrula
    for company_id in final_results["companies"]:
        if any(c["id"] == company_id for c in companies):
            valid_results["companies"].append(company_id)
    
    return valid_results

while True:
    print("\n\n-------------------------------")
    query = input("Sorgunuzu girin (çıkmak için q): ")
    print("\n\n")
    
    if query.lower() == "q":
        break
    
    result = enhanced_search(query)
    print(f"Sonuç: {result}")
    
    # Bu ID'lerin neye karşılık geldiğini netlik için göster
    if result.get("cities"):
        print("\nŞehirler:")
        for city_id in result["cities"]:
            city = next((c for c in cities if c["id"] == city_id), None)
            if city:
                print(f"  ID: {city_id}, İsim: {city['city']}")
    
    if result.get("tax"):
        print("\nVergi Daireleri:")
        for tax_id in result["tax"]:
            office = next((t for t in tax_offices if t["id"] == tax_id), None)
            if office:
                city = next((c for c in cities if c["id"] == office.get("city_id")), {"city": "Bilinmiyor"})
                print(f"  ID: {tax_id}, İsim: {office['name']}, Şehir: {city['city']}")
                
    if result.get("companies"):
        print("\nFirmalar:")
        for company_id in result["companies"]:
            company = next((c for c in companies if c["id"] == company_id), None)
            if company:
                city = next((c for c in cities if c["id"] == company.get("city_id")), {"city": "Bilinmiyor"})
                tax_office = {"name": "Belirtilmemiş"}
                if company.get("tax_id"):
                    tax_office = next((t for t in tax_offices if t["id"] == company.get("tax_id")), {"name": "Belirtilmemiş"})
                print(f"  ID: {company_id}, İsim: {company['name']}, Şehir: {city['city']}, Vergi Dairesi: {tax_office['name']}")