"""
AGPC Chatbot — Smart RAG Indexer
Run this ONCE to build the vector index from your JSON data files.
Usage: python3 build_index.py
"""

import json
import os
import sys

# ─── CONFIG ───────────────────────────────────────────────────────────────────
JSON_FILE_1 = "agpc_chatbot.json"       # Main KB with intents + FAQ
JSON_FILE_2 = "agpc_scraped_v2.json"    # Detailed scraped data
DB_PATH     = "./agpc_chroma_db"        # Where ChromaDB will be stored
MODEL_NAME  = "all-MiniLM-L6-v2"       # Free local embedding model
# ─────────────────────────────────────────────────────────────────────────────


def load_json(path):
    if not os.path.exists(path):
        print(f"  ⚠  File not found: {path} — skipping")
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def extract_chunks_from_kb(data: dict) -> list[dict]:
    """Extract semantic chunks from agpc_chatbot.json"""
    chunks = []

    # ── Institution core facts ──────────────────────────────────────────────
    inst = data.get("institution", {})
    addr = inst.get("address", {})
    chunks.append({
        "id": "inst_core",
        "text": (
            f"Arasan Ganesan Polytechnic College (AGPC) was founded in {inst.get('founded', 1981)}. "
            f"It is located at {addr.get('street', '')}, {addr.get('city', 'Sivakasi')}, "
            f"Tamil Nadu – {addr.get('pincode', '626130')}. "
            f"Phone: {inst.get('phone', ['+91 95002 99595'])[0]}. "
            f"Website: {inst.get('website', 'https://www.arasanganesanpoly.edu.in')}. "
            f"Affiliated to AICTE and DOTE Tamil Nadu. 45 years of excellence (as of 2026). "
            f"Named after Founding Correspondent Arasan A.M.S.G. Vijayakumar."
        ),
        "category": "institution",
    })

    # ── Emails ──────────────────────────────────────────────────────────────
    emails = inst.get("email", {})
    chunks.append({
        "id": "inst_contact",
        "text": (
            f"AGPC Contact Information: "
            f"Admissions email: {emails.get('admissions', 'admission@arasanganesanpoly.edu.in')}. "
            f"Principal email: {emails.get('principal', 'principal@arasanganesanpoly.edu.in')}. "
            f"General email: {emails.get('general', 'agpoly1981@gmail.com')}. "
            f"Phone: +91 95002 99595. "
            f"Website: https://www.arasanganesanpoly.edu.in."
        ),
        "category": "contact",
    })

    # ── Management ──────────────────────────────────────────────────────────
    mgmt = data.get("management", {})
    principal = mgmt.get("principal", {})
    chairman = mgmt.get("chairman", {})
    correspondent = mgmt.get("correspondent", {})
    chunks.append({
        "id": "management",
        "text": (
            f"AGPC Management: "
            f"Principal: {principal.get('name', 'Dr. M. Nandakumar')} "
            f"({principal.get('qualification', 'M.E., Ph.D.')}), specializing in Printing Engineering. "
            f"Chairman: {chairman.get('name', 'Thiru. G. Ashokan')} ({chairman.get('qualification', 'B.Com.')}). "
            f"Correspondent: {correspondent.get('name', 'Thiru. A. Ganeshkumar')} "
            f"({correspondent.get('qualification', 'B.Tech., M.B.A.')}). "
            f"Principal has authored textbooks, published papers at national/international conferences, "
            f"and led AGPC to 300+ industry tie-ups."
        ),
        "category": "management",
    })

    # ── Departments (one chunk each) ─────────────────────────────────────────
    depts = data.get("departments", {})
    for dept_id, dept in depts.items():
        parts = [f"Department: {dept.get('name', dept_id)}."]
        if dept.get("established"):
            parts.append(f"Established: {dept['established']}.")
        if dept.get("intake"):
            parts.append(f"Annual intake: {dept['intake']} students.")
        if dept.get("focus"):
            parts.append(f"Focus: {dept['focus']}.")
        if dept.get("placement_rate"):
            parts.append(f"Placement rate: {dept['placement_rate']}.")
        if dept.get("lab_value"):
            parts.append(f"Lab value: {dept['lab_value']}.")
        if dept.get("machinery_value"):
            parts.append(f"Machinery value: {dept['machinery_value']}.")
        if dept.get("board_ranks"):
            parts.append(f"Academic rank: {dept['board_ranks']}.")
        if dept.get("special_labs"):
            parts.append(f"Special labs: {', '.join(dept['special_labs'])}.")
        if dept.get("emerging_tech"):
            parts.append(f"Emerging tech: {', '.join(dept['emerging_tech'])}.")
        if dept.get("career_roles"):
            parts.append(f"Career roles: {', '.join(dept['career_roles'][:4])}.")
        chunks.append({
            "id": f"dept_{dept_id}",
            "text": " ".join(parts),
            "category": "department",
        })

    # ── Admissions ──────────────────────────────────────────────────────────
    adm = data.get("admissions", {})
    eligibility = adm.get("eligibility", {})
    channels = adm.get("channels", {})
    govt = channels.get("government_quota", "")
    mgmt_q = channels.get("management_quota", "")

    chunks.append({
        "id": "admissions_overview",
        "text": (
            f"AGPC Admissions: Currently {adm.get('status', 'OPEN for 2026-2027')}. "
            f"Governed by DOTE Tamil Nadu. No capitation fee or hidden charges. "
            f"Eligibility (regular): {eligibility.get('regular_first_year', 'SSLC/10th with Maths & Science')}. "
            f"Lateral entry: {eligibility.get('lateral_entry', 'HSC or 2-year ITI holders')}. "
            f"Government quota: apply via TN Govt online counselling. "
            f"Management quota: apply directly — email admission@arasanganesanpoly.edu.in or call +91 95002 99595."
        ),
        "category": "admissions",
    })

    # Scholarships
    scholarships = adm.get("scholarships", [])
    if scholarships:
        chunks.append({
            "id": "scholarships",
            "text": (
                "AGPC Scholarships and fee concessions: "
                + "; ".join(scholarships[:8])
                + ". College actively helps ALL eligible students apply."
            ),
            "category": "admissions",
        })

    # Documents required
    docs = adm.get("documents_required", {})
    all_docs = docs.get("all", [])
    if all_docs:
        chunks.append({
            "id": "admission_docs",
            "text": (
                "Documents required for AGPC admission: "
                + ", ".join(all_docs)
                + ". Additional: Income Certificate for scholarship applicants. "
                + "For lateral entry: ITI Certificate or HSC Vocational Certificate."
            ),
            "category": "admissions",
        })

    # ── Schemes ──────────────────────────────────────────────────────────────
    schemes = data.get("schemes", {})
    for scheme_id, scheme in schemes.items():
        activities = scheme.get("activities", []) or scheme.get("major_activities", [])
        benefits = (
            scheme.get("student_benefits", [])
            or scheme.get("benefits_to_students", [])
            or scheme.get("support", [])
        )
        trades = scheme.get("trades_offered", [])
        parts = [f"Scheme: {scheme.get('full_name', scheme_id)}."]
        parts.append(f"Type: {scheme.get('type', '')}.")
        parts.append(f"Goal: {scheme.get('goal', '')}.")
        if activities:
            parts.append(f"Activities: {', '.join(activities[:3])}.")
        if benefits:
            parts.append(f"Benefits: {', '.join(benefits[:3])}.")
        if trades:
            parts.append(f"Trades offered: {', '.join(trades)}.")
        chunks.append({
            "id": f"scheme_{scheme_id}",
            "text": " ".join(parts),
            "category": "scheme",
        })

    # ── Clubs ────────────────────────────────────────────────────────────────
    clubs = data.get("clubs", [])
    for i, club in enumerate(clubs):
        acts = club.get("activities", [])
        bens = club.get("benefit", "")
        chunks.append({
            "id": f"club_{i}",
            "text": (
                f"Club: {club.get('name', '')} — {club.get('full_name', '')}. "
                f"Type: {club.get('type', '')}. "
                f"Activities: {', '.join(acts[:4])}. "
                f"Benefit: {bens}."
            ),
            "category": "club",
        })

    # ── Facilities ──────────────────────────────────────────────────────────
    fac = data.get("facilities", {})

    # Library
    lib = fac.get("library_clair", {})
    col = lib.get("collection", {})
    chunks.append({
        "id": "facility_library",
        "text": (
            f"CLAIR (Centre for Library and Information Resources) at AGPC: "
            f"Named after Founding Correspondent Arasan A.M.S.G. Vijayakumar. "
            f"Size: {lib.get('size_sqm', 540)} sq.m. "
            f"Collection: {col.get('books', 31189)} books, {col.get('book_bank', 1453)} book-bank, "
            f"{col.get('cds', 2036)} CDs, {col.get('periodicals', 30)} periodicals, "
            f"{col.get('newspapers', 4)} daily newspapers. "
            f"Digital library: D-Space at http://103.104.68.42:8080/agpc/. OPAC available. "
            f"Timings: Lunch 12:35–1:10 PM, Evening 4:15–5:15 PM. "
            f"Overdue fine: Rs. 1/day. Books returnable within 2 weeks. "
            f"Journals cannot be borrowed. Silence mandatory. "
            f"Librarian: Dr. C. Ramasubramanian (M.C.A., M.E., M.L.I.Sc., SET, Ph.D.)."
        ),
        "category": "facility",
    })

    # Hostels
    host = fac.get("hostels", {})
    chunks.append({
        "id": "facility_hostel",
        "text": (
            "AGPC Hostels: Separate hostels for BOYS and GIRLS. "
            "Features: spacious ventilated rooms, 24/7 security & supervision, "
            "nutritious hygienic mess food (3 meals/day), study rooms, "
            "RO drinking water, indoor games, supportive wardens."
        ),
        "category": "facility",
    })

    # Transport
    trans = fac.get("transport", {})
    routes = trans.get("routes", [])
    route_str = "; ".join(
        [f"Bus {r.get('bus', '')}: {r.get('from', '')} → {r.get('to', 'AGPC')}" for r in routes]
        if routes else ["Sivakasi → AGPC", "Srivilliputhur → AGPC", "Sattur → AGPC"]
    )
    chunks.append({
        "id": "facility_transport",
        "text": (
            f"AGPC Transport: {trans.get('buses', 3)} college buses. "
            f"Routes: {route_str}. "
            f"Available for both staff and students. "
            f"Also used for industrial visits, educational tours, and inter-college tournaments."
        ),
        "category": "facility",
    })

    # Sports
    pe = fac.get("physical_education", {})
    outdoor = pe.get("outdoor_sports", [])
    indoor = pe.get("indoor_sports", [])
    chunks.append({
        "id": "facility_sports",
        "text": (
            f"AGPC Physical Education & Sports: "
            f"400m athletics track. "
            f"Outdoor sports ({len(outdoor)}): {', '.join(outdoor)}. "
            f"Indoor sports ({len(indoor)}): {', '.join(indoor)}. "
            f"Compulsory evening games for all students. "
            f"Latest: {pe.get('sports_day', '44th Annual Sports Day 2025-2026')}."
        ),
        "category": "facility",
    })

    # ── Placement cell ───────────────────────────────────────────────────────
    pc = data.get("placement_cell", {})
    sectors = pc.get("sectors_hiring", [])
    roles = pc.get("job_roles", [])
    modules = pc.get("training_modules", [])
    chunks.append({
        "id": "placement",
        "text": (
            f"AGPC Training & Placement Cell: "
            f"Mission: {pc.get('mission', '100% placement assistance')}. "
            f"Training modules: {', '.join(modules[:5])}. "
            f"Recruitment types: On-Campus, Off-Campus, Pool Campus, Apprenticeship. "
            f"Sectors hiring: {', '.join(sectors[:5])}. "
            f"Common roles: {', '.join(roles[:5])}. "
            f"Printing Technology: 100% placement consistently (20+ batches, India & abroad). "
            f"EEE: 100% placement in 2025."
        ),
        "category": "placement",
    })

    # ── Exam cell ────────────────────────────────────────────────────────────
    ec = data.get("examination_cell", {})
    sched = ec.get("current_schedule_2026", {})
    chunks.append({
        "id": "exam_cell",
        "text": (
            f"AGPC Examination Cell: Governed by DOTE Tamil Nadu. "
            f"Officer: {ec.get('officer', 'Dr. C. Ramasubramanian')}. "
            f"April 2026 Theory Exam starts: {sched.get('theory', '23 March 2026')}. "
            f"April 2026 Practical Exam starts: {sched.get('practical', '30 March 2026')}."
        ),
        "category": "exam",
    })

    # ── FAQ pairs (each is a perfect standalone chunk) ───────────────────────
    for i, faq in enumerate(data.get("faq_pairs", [])):
        chunks.append({
            "id": f"faq_{i}",
            "text": f"Q: {faq['q']} A: {faq['a']}",
            "category": "faq",
        })

    # ── Key numbers ──────────────────────────────────────────────────────────
    kn = data.get("key_numbers", {})
    chunks.append({
        "id": "key_numbers",
        "text": (
            f"AGPC Key Facts: {kn.get('industry_tie_ups', '300+')} industry tie-ups, "
            f"{kn.get('departments', 7)} departments, {kn.get('clubs', 11)} clubs, "
            f"{kn.get('schemes', 4)} schemes, {kn.get('facilities', 6)} facilities. "
            f"Basic Engineering intake: {kn.get('basic_eng_intake', 420)}/year. "
            f"Civil Engineering intake: {kn.get('civil_eng_intake', 60)}/year. "
            f"ECE lab value: {kn.get('ece_lab_value_inr', 'Rs. 75 Lakhs')}. "
            f"Printing machinery: {kn.get('printing_machinery_value_inr', 'Rs. 2 Crores+')}. "
            f"Library: {kn.get('library_books', 31189)} books. "
            f"Buses: {kn.get('buses', 3)}. "
            f"Sports Day: {kn.get('sports_day_edition', '44th (2025-2026)')}."
        ),
        "category": "stats",
    })

    # ── Governing council ────────────────────────────────────────────────────
    gc = data.get("governing_council", [])
    if gc:
        names = [f"{m['no']}. {m['name']} ({m['role']})" for m in gc[:6]]
        chunks.append({
            "id": "governing_council",
            "text": (
                "AGPC Governing Council (11 members): "
                + "; ".join(names)
                + "; plus 5 government/AICTE nominees."
            ),
            "category": "management",
        })

    return chunks


def extract_chunks_from_scraped(data: dict) -> list[dict]:
    """Extract additional chunks from agpc_scraped_v2.json (deduplication safe)"""
    chunks = []

    # Motto + vision + mission
    mv = data.get("mission_vision_motto", {})
    if mv:
        mission_pts = mv.get("mission_points", [])
        chunks.append({
            "id": "motto_vision",
            "text": (
                f"AGPC Motto: '{mv.get('motto', '')}'. "
                f"Vision: {mv.get('vision', '')}. "
                f"Mission highlights: {'; '.join(mission_pts[:4])}. "
                f"Why students choose AGPC: {'; '.join(mv.get('why_students_choose_agpc', [])[:5])}."
            ),
            "category": "institution",
        })

    # Emblem
    emb = data.get("emblem_symbolism", {})
    if emb:
        chunks.append({
            "id": "emblem",
            "text": (
                f"AGPC Emblem: A WHEEL. "
                f"Centre = AGPC as the driving force of knowledge. "
                f"Spokes = rays of knowledge spreading in all directions. "
                f"Arrowheads = focus, direction, power of education. "
                f"Overall meaning: {emb.get('overall_meaning', '')}."
            ),
            "category": "institution",
        })

    # Founding trust
    ft = data.get("founding_trust", {})
    if ft:
        chunks.append({
            "id": "founding_trust",
            "text": (
                f"AGPC Founding Trust: Vision: {ft.get('vision', '')}. "
                f"Founding Correspondent: {ft.get('founding_correspondent', {}).get('name', 'Arasan A.M.S.G. Vijayakumar')}. "
                f"Core values: {', '.join(ft.get('core_values', [])[:3])}."
            ),
            "category": "institution",
        })

    # Higher education guidance
    pc2 = data.get("training_and_placement_cell", {})
    he = pc2.get("higher_education_support", [])
    if he:
        chunks.append({
            "id": "higher_studies",
            "text": (
                "Higher studies after AGPC diploma: "
                + "; ".join(he)
                + ". Students can pursue lateral entry to B.E./B.Tech via TNEA counselling."
            ),
            "category": "placement",
        })

    # Entrepreneurship support
    ent = pc2.get("entrepreneurship_support", [])
    if ent:
        chunks.append({
            "id": "entrepreneurship",
            "text": (
                "AGPC supports entrepreneurship in: "
                + ", ".join(ent)
                + ". Supported through CIICP and CDTP schemes."
            ),
            "category": "placement",
        })

    # FAQ from scraped data (avoid dupe IDs by using prefix)
    for i, faq in enumerate(data.get("faq", [])):
        chunks.append({
            "id": f"faq2_{i}",
            "text": f"Q: {faq['question']} A: {faq['answer']}",
            "category": "faq",
        })

    return chunks


def build_index(
    json1=JSON_FILE_1,
    json2=JSON_FILE_2,
    db_path=DB_PATH,
    model_name=MODEL_NAME,
):
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("❌ Missing packages. Run: pip install chromadb sentence-transformers --break-system-packages")
        sys.exit(1)

    print("\n🔨  AGPC Chatbot — Building Knowledge Index")
    print("=" * 55)

    # Load data
    print(f"📂  Loading {json1} …")
    data1 = load_json(json1)
    print(f"📂  Loading {json2} …")
    data2 = load_json(json2)

    # Extract chunks
    chunks1 = extract_chunks_from_kb(data1) if data1 else []
    chunks2 = extract_chunks_from_scraped(data2) if data2 else []
    all_chunks = chunks1 + chunks2

    # Deduplicate by ID
    seen = set()
    unique_chunks = []
    for c in all_chunks:
        if c["id"] not in seen:
            seen.add(c["id"])
            unique_chunks.append(c)

    print(f"📊  Total chunks to index: {len(unique_chunks)}")

    # Embedding model
    print(f"\n🧠  Loading embedding model ({model_name}) …")
    model = SentenceTransformer(model_name)

    # ChromaDB
    print(f"🗄️   Initializing ChromaDB at {db_path} …")
    client = chromadb.PersistentClient(path=db_path)
    try:
        client.delete_collection("agpc_knowledge")
        print("   ♻️  Cleared old index")
    except Exception:
        pass

    collection = client.create_collection(
        name="agpc_knowledge",
        metadata={"hnsw:space": "cosine"},
    )

    # Batch embed + store
    print("\n⚡  Embedding and indexing …")
    texts = [c["text"] for c in unique_chunks]
    ids = [c["id"] for c in unique_chunks]
    metas = [{"category": c["category"]} for c in unique_chunks]

    if texts:
        # Embed in batches of 64
        all_embeddings = []
        batch_size = 64
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embs = model.encode(batch, show_progress_bar=False).tolist()
            all_embeddings.extend(embs)
            print(f"   Embedded {min(i + batch_size, len(texts))}/{len(texts)}", end="\r")

        collection.add(
            documents=texts,
            embeddings=all_embeddings,
            ids=ids,
            metadatas=metas,
        )
    else:
        print("   ⚠️  No chunks to index. Skipping ChromeDB insertion.")

    print(f"\n\n✅  Index built! {len(unique_chunks)} chunks stored in ChromaDB.")
    print(f"    Path: {os.path.abspath(db_path)}")
    print("\n📋  Chunk categories:")
    from collections import Counter
    cat_count = Counter(c["category"] for c in unique_chunks)
    for cat, count in sorted(cat_count.items(), key=lambda x: -x[1]):
        print(f"    {cat:<20} {count} chunks")
    print("\n🚀  Run app.py to start the chatbot!")


if __name__ == "__main__":
    build_index()
