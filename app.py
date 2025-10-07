import os
from datetime import datetime, timedelta
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy import func
from sqlalchemy.orm import joinedload
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session, send_file
from collections import defaultdict
import io
from decimal import Decimal, InvalidOperation







BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, "app.db")

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret")

# Usa Postgres in produzione se è impostata DATABASE_URL, altrimenti SQLite locale
DB_URL = os.environ.get("DATABASE_URL")
if DB_URL:
    # Normalizza eventuale prefisso postgres://
    if DB_URL.startswith("postgres://"):
        DB_URL = DB_URL.replace("postgres://", "postgresql+psycopg://", 1)
    elif DB_URL.startswith("postgresql://"):
        DB_URL = DB_URL.replace("postgresql://", "postgresql+psycopg://", 1)
    app.config["SQLALCHEMY_DATABASE_URI"] = DB_URL
else:
    app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{DB_PATH}"


app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"

# ---------- MODELS ----------
class User(UserMixin, db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False, default="client")  # 'admin' or 'client'
    display_name = db.Column(db.String(120), nullable=False)

    # NEW: per gli employee
    works_for_client_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    access_sala = db.Column(db.Boolean, default=True)
    access_cucina = db.Column(db.Boolean, default=True)

    # NUOVO: livello permessi per admin: 'full' | 'partial' (None per non-admin)
    admin_level = db.Column(db.String(20), nullable=True, default="full")

    def set_password(self, password: str):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

class Department(db.Model):
    __tablename__ = "departments"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False, unique=True)
    active = db.Column(db.Boolean, default=True)
    macro_area = db.Column(db.String(20), nullable=False, default="sala")  # 'sala' | 'cucina'

    products = db.relationship("Product", back_populates="department", lazy=True)

class Product(db.Model):
    __tablename__ = "products"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(180), nullable=False)
    # sku rimosso dall'UI: il campo può esistere nel DB ma non viene usato
    sku = db.Column(db.String(80), nullable=True)
    active = db.Column(db.Boolean, default=True)

    department_id = db.Column(db.Integer, db.ForeignKey("departments.id"), nullable=False)
    department = db.relationship("Department", back_populates="products")

class RequestHeader(db.Model):
    __tablename__ = "requests"
    id = db.Column(db.Integer, primary_key=True)
    client_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    status = db.Column(db.String(20), nullable=False, default="bozza")  # bozza | inviata | evasa
    # NEW: chi l’ha inviata (employee)
    submitted_by_user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    submitted_by = db.relationship("User", foreign_keys=[submitted_by_user_id])

    client = db.relationship("User", foreign_keys=[client_id])  # niente backref ambiguo
    items = db.relationship("RequestItem", back_populates="request", cascade="all, delete-orphan", lazy=True)

class RequestItem(db.Model):
    __tablename__ = "request_items"
    id = db.Column(db.Integer, primary_key=True)
    request_id = db.Column(db.Integer, db.ForeignKey("requests.id"), nullable=False)
    product_id = db.Column(db.Integer, db.ForeignKey("products.id"), nullable=False)
    note = db.Column(db.String(255), nullable=True)

    request = db.relationship("RequestHeader", back_populates="items")
    product = db.relationship("Product")
    qty_requested = db.Column(db.Numeric(10,3), nullable=False, default=0)

class DistributionPlan(db.Model):
    __tablename__ = "distribution_plans"
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    created_by_user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    status = db.Column(db.String(20), nullable=False, default="bozza")  # bozza | confermato

    created_by = db.relationship("User")
    lines = db.relationship("DistributionLine", back_populates="plan", cascade="all, delete-orphan", lazy=True)

class ClientProductRecommendation(db.Model):
    __tablename__ = "client_product_recommendations"
    id = db.Column(db.Integer, primary_key=True)
    client_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    product_id = db.Column(db.Integer, db.ForeignKey("products.id"), nullable=False)
    recommended_qty = db.Column(db.Integer, nullable=False, default=0)

    __table_args__ = (
        db.UniqueConstraint("client_id", "product_id", name="uq_client_product_rec"),
    )

from decimal import Decimal

class DistributionLine(db.Model):
    __tablename__ = "distribution_lines"
    id = db.Column(db.Integer, primary_key=True)
    plan_id = db.Column(db.Integer, db.ForeignKey("distribution_plans.id"), nullable=False)
    client_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    product_id = db.Column(db.Integer, db.ForeignKey("products.id"), nullable=False)
    # CAMBIA QUI: Numeric fixed point 2 decimali
    qty_in  = db.Column(db.Numeric(10,3), nullable=False, default=0)
    qty_out = db.Column(db.Numeric(10,3), nullable=False, default=0)

    plan = db.relationship("DistributionPlan", back_populates="lines")
    client = db.relationship("User")
    product = db.relationship("Product")

class MacroAreaAccess(db.Model):
    __tablename__ = "macro_area_access"
    id = db.Column(db.Integer, primary_key=True)
    area = db.Column(db.String(20), nullable=False, unique=True)  # 'sala' | 'cucina'
    code = db.Column(db.String(80), nullable=True, default="")    # vuoto = nessun codice richiesto

class UserMacroAccess(db.Model):
    __tablename__ = "user_macro_access"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    area = db.Column(db.String(50), nullable=False)

    __table_args__ = (db.UniqueConstraint("user_id", "area", name="uq_user_area"),)

class MacroAreaCodes(db.Model):
    __tablename__ = "macro_area_codes"
    id = db.Column(db.Integer, primary_key=True)
    sala_code = db.Column(db.String(50), nullable=True)
    cucina_code = db.Column(db.String(50), nullable=True)

class AppSetting(db.Model):
    __tablename__ = "app_settings"
    key = db.Column(db.String(120), primary_key=True)
    value = db.Column(db.String(500), nullable=True)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ---------- UTILS ----------
def ensure_seed_data():
    """Crea un utente admin predefinito e alcuni dati demo se il DB è vuoto."""
    if User.query.count() == 0:
        # Admin
        admin = User(username="admin", display_name="Admin", role="admin", admin_level="full")
        admin.set_password("admin")
        db.session.add(admin)

        # Negozi (role=client) – password casuale non usabile
        c1 = User(username="shop1", display_name="Negozio 1", role="client")
        c1.set_password(os.urandom(16).hex())
        c2 = User(username="shop2", display_name="Negozio 2", role="client")
        c2.set_password(os.urandom(16).hex())
        db.session.add_all([c1, c2])
        db.session.flush()  # per avere c1.id/c2.id

        # Un utente/employee di test legato a Negozio 1
        e1 = User(
            username="utente1",
            display_name="Mario (Negozio 1)",
            role="employee",
            works_for_client_id=c1.id,
            access_sala=True,
            access_cucina=True
        )
        e1.set_password("utente1")
        db.session.add(e1)

        # Reparti + prodotti di esempio
        d1 = Department(name="Drogheria", macro_area="sala")
        d2 = Department(name="Freschi",   macro_area="cucina")
        db.session.add_all([d1, d2])

        p1 = Product(name="Pasta 500g", department=d1)
        p2 = Product(name="Passata 700ml", department=d1)
        p3 = Product(name="Latte 1L", department=d2)
        db.session.add_all([p1, p2, p3])

        db.session.commit()

            # Crea record per codici macro-area se mancanti
        if MacroAreaAccess.query.filter_by(area="sala").first() is None:
            db.session.add(MacroAreaAccess(area="sala", code=""))  # vuoto = non serve codice
        if MacroAreaAccess.query.filter_by(area="cucina").first() is None:
            db.session.add(MacroAreaAccess(area="cucina", code=""))
        db.session.commit()

from sqlalchemy import inspect, text

def ensure_schema_upgrade():
    insp = inspect(db.engine)
    cols = {c["name"] for c in insp.get_columns("users")}
    if "admin_level" not in cols:
        # Aggiungi la colonna con default 'full' (compatibile con SQLite e Postgres)
        with db.engine.begin() as conn:
            conn.execute(text("ALTER TABLE users ADD COLUMN admin_level VARCHAR(20) DEFAULT 'full'"))
            # backfill: admin -> full, altri -> none (o lasciali default)
            conn.execute(text("UPDATE users SET admin_level='full' WHERE role='admin'"))
            conn.execute(text("UPDATE users SET admin_level='none' WHERE role!='admin'"))
    try:
        if db.engine.name.startswith("postgresql"):
            cols = {c["name"]: c for c in insp.get_columns("request_items")}
            col = cols.get("qty_requested")
            # se la colonna non è già numeric, la trasformo
            if col and "numeric" not in str(col.get("type", "")).lower():
                with db.engine.begin() as conn:
                    conn.execute(text("""
                        ALTER TABLE request_items
                        ALTER COLUMN qty_requested
                        TYPE NUMERIC(10,3)
                        USING qty_requested::numeric
                    """))
    except Exception as _e:
        # logga se vuoi, ma non bloccare l’avvio
        pass

    cols_req = {c["name"] for c in insp.get_columns("requests")}
    if "submitted_by_user_id" not in cols_req:
        with db.engine.begin() as conn:
            conn.execute(text("ALTER TABLE requests ADD COLUMN submitted_by_user_id INTEGER"))

def get_or_create_draft_request(client_id: int) -> RequestHeader:
    draft = RequestHeader.query.filter_by(client_id=client_id, status="bozza").order_by(RequestHeader.created_at.desc()).first()
    if draft is None:
        draft = RequestHeader(client_id=client_id, status="bozza")
        db.session.add(draft)
        db.session.commit()
    return draft

def get_or_create_draft_plan(admin_id: int) -> "DistributionPlan":
    plan = (DistributionPlan.query
            .filter_by(created_by_user_id=admin_id, status="bozza")
            .order_by(DistributionPlan.created_at.desc())
            .first())
    if plan is None:
        plan = DistributionPlan(created_by_user_id=admin_id, status="bozza")
        db.session.add(plan)
        db.session.commit()
    return plan
 
def combined_client_items_for_date(client_id: int, date_str: str):
    """
    Prodotti delle giacenze 'inviata' in quel giorno (Europe/Rome), raggruppati per macro area.
    """
    start_utc, end_utc = local_day_range_utc(date_str)

    reqs = (
        RequestHeader.query
        .filter(RequestHeader.client_id == client_id, RequestHeader.status == "inviata")
        .filter(RequestHeader.created_at >= start_utc)
        .filter(RequestHeader.created_at <  end_utc)
        .all()
    )

    qty_by_pid = defaultdict(int)
    product_by_pid = {}
    for rh in reqs:
        for it in rh.items:
            q = max(0, it.qty_requested or 0)
            if q <= 0:
                continue
            qty_by_pid[it.product_id] += q
            if it.product_id not in product_by_pid:
                product_by_pid[it.product_id] = it.product

    groups = {"sala": [], "cucina": []}
    for pid, total in qty_by_pid.items():
        prod = product_by_pid.get(pid)
        if not prod:
            continue
        area = (prod.department.macro_area or "sala")
        groups[area].append({"product": prod, "qty": total})

    for area in groups:
        groups[area].sort(
            key=lambda r: (
                (r["product"].department.name or "").lower(),
                (r["product"].name or "").lower(),
            )
        )
    return groups


def require_role(role):
    def wrapper(fn):
        from functools import wraps
        @wraps(fn)
        def decorated(*args, **kwargs):
            if not current_user.is_authenticated:
                return login_manager.unauthorized()
            if current_user.role != role:
                flash("Non autorizzato.", "danger")
                return redirect(url_for("index"))
            return fn(*args, **kwargs)
        return decorated
    return wrapper

def require_roles(*roles):
    def wrapper(fn):
        from functools import wraps
        @wraps(fn)
        def decorated(*args, **kwargs):
            if not current_user.is_authenticated:
                return login_manager.unauthorized()
            if current_user.role not in roles:
                flash("Non autorizzato.", "danger")
                return redirect(url_for("index"))
            return fn(*args, **kwargs)
        return decorated
    return wrapper
def current_shop_id():
    """Se è employee, ritorna il negozio assegnato; se è 'client' usa il suo id."""
    if current_user.role == "employee":
        return current_user.works_for_client_id
    return current_user.id

def get_macro_code(area: str) -> str:
    """Codice richiesto per l'area ('sala'|'cucina'); '' = nessun codice richiesto."""
    area = (area or "").strip().lower()
    rec = MacroAreaAccess.query.filter_by(area=area).first()
    return rec.code if rec and rec.code is not None else ""

def set_macro_code(area: str, code: str):
    area = (area or "").strip().lower()
    rec = MacroAreaAccess.query.filter_by(area=area).first()
    if rec is None:
        rec = MacroAreaAccess(area=area, code=(code or ""))
        db.session.add(rec)
    else:
        rec.code = (code or "")
    db.session.commit()


# --- HELPER per pairing Sala/Cucina nel giorno selezionato --------------------
from collections import defaultdict  # se non l'hai già importato sopra

def is_partial_admin(user=None) -> bool:
    u = user or current_user
    return bool(u and getattr(u, "role", None) == "admin" and getattr(u, "admin_level", "full") == "partial")
# --- Date helpers: filtri per giorno locale Europe/Rome su campi UTC ---

def list_macro_areas() -> list:
    """Tutte le macro-aree definite (ordinate)."""
    return [m.area for m in MacroAreaAccess.query.order_by(MacroAreaAccess.area.asc()).all()]

def employee_allowed_areas(u: User) -> set:
    """Macro-aree consentite a un employee (dinamico); fallback ai 2 booleani legacy."""
    rows = UserMacroAccess.query.filter_by(user_id=u.id).all()
    if rows:
        return {r.area for r in rows}
    allowed = set()
    if getattr(u, "access_sala", False): allowed.add("sala")
    if getattr(u, "access_cucina", False): allowed.add("cucina")
    return allowed

def parse_decimal(val, default=Decimal("0")) -> Decimal:
    if val is None:
        return default
    # accetta 0.5 e 0,5
    s = str(val).strip().replace(',', '.')
    try:
        d = Decimal(s)
    except (InvalidOperation, ValueError, TypeError):
        return default
    return d if d >= 0 else Decimal("0")

# app.py
from decimal import Decimal, InvalidOperation

@app.template_filter('fmt_qty')
def fmt_qty(val):
    """
    Mostra 3 -> "3", 3.0 -> "3", 3.50 -> "3.5", 3.25 -> "3.25".
    Niente zeri inutili, niente notazione scientifica.
    """
    if val is None:
        return "0"
    # prova Decimal per preservare i decimali
    try:
        d = Decimal(str(val))
        if d == d.to_integral_value():
            return str(int(d))
        # altrimenti togli zeri in coda
        s = format(d.normalize())
        # format(normalize) a volte può dare "0E-7": riformattiamo in fixed e ripuliamo
        if "E" in s or "e" in s:
            s = ("{:.6f}".format(d)).rstrip('0').rstrip('.')
        return s
    except (InvalidOperation, ValueError):
        # fallback float
        try:
            f = float(val)
        except Exception:
            return str(val)
        if abs(f - round(f)) < 1e-9:
            return str(int(round(f)))
        return ("{:.6f}".format(f)).rstrip('0').rstrip('.')


from datetime import timezone
from dateutil import tz
TZ_ROME = tz.gettz("Europe/Rome")

def to_rome(dt):
    if not dt: return None
    return dt.replace(tzinfo=timezone.utc).astimezone(TZ_ROME)

@app.template_filter('local_dt')
def local_dt(val, fmt='%Y-%m-%d %H:%M'):
    dt = to_rome(val)
    return dt.strftime(fmt) if dt else ''

@app.template_filter('local_time')
def local_time(val, fmt='%H:%M'):
    dt = to_rome(val)
    return dt.strftime(fmt) if dt else ''


def _local_ymd(dt: datetime) -> str:
    """Restituisce 'YYYY-MM-DD' in fuso Europe/Rome a partire da un dt UTC (naive)."""
    if dt is None:
        return ""
    if TZ_ROME is not None:
        # created_at è salvato naive in UTC → marcala come UTC e converti
        dt = dt.replace(tzinfo=timezone.utc)
        dt_local = dt.astimezone(TZ_ROME)
    else:
        dt_local = dt
    return dt_local.strftime("%Y-%m-%d")

def build_request_rows_for_admin(only_date: str = None):
    """
    Ritorna la lista di righe per 'Giacenze ricevute'.
    Se only_date è 'YYYY-MM-DD', filtra solo quel giorno (Europe/Rome),
    altrimenti mostra tutte le giacenze, ordinate dalla più recente.
    """
    clients = (User.query
               .filter_by(role="client")
               .order_by(User.display_name)
               .all())

    rows = []
    for c in clients:
        # tutte le 'inviata' del client (con join per sapere la macro-area degli item)
        reqs = (
            RequestHeader.query
            .filter_by(client_id=c.id, status="inviata")
            .options(
                joinedload(RequestHeader.items)
                .joinedload(RequestItem.product)
                .joinedload(Product.department)
            )
            .order_by(RequestHeader.created_at.asc())  # cronologico ↑ per pairing consistente
            .all()
        )

        # raggruppa per giorno locale
        by_day = {}  # ymd -> {"sala":[{id,created_at}], "cucina":[...]}
        for rh in reqs:
            ymd = _local_ymd(rh.created_at)
            if only_date and ymd != only_date:
                continue

            has_sala = any(
                (it.qty_requested or 0) > 0
                and it.product and it.product.department
                and (it.product.department.macro_area or "sala") == "sala"
                for it in rh.items
            )
            has_cucina = any(
                (it.qty_requested or 0) > 0
                and it.product and it.product.department
                and (it.product.department.macro_area or "sala") == "cucina"
                for it in rh.items
            )

            slot = by_day.setdefault(ymd, {"sala": [], "cucina": []})
            if has_sala:
                slot["sala"].append({"request_id": rh.id, "created_at": rh.created_at})
            if has_cucina:
                slot["cucina"].append({"request_id": rh.id, "created_at": rh.created_at})

        # pairing per indice all’interno dello stesso giorno
        for ymd, lists in by_day.items():
            sala_list = lists["sala"]
            cucina_list = lists["cucina"]
            n = max(len(sala_list), len(cucina_list))
            for i in range(n):
                areas = {}
                if i < len(sala_list):   areas["sala"]   = sala_list[i]
                if i < len(cucina_list): areas["cucina"] = cucina_list[i]
                created_at = max(a["created_at"] for a in areas.values())
                rows.append({
                    "client_id":   c.id,
                    "client_name": c.display_name,
                    "rank":        i + 1,
                    "created_at":  created_at,
                    "is_complete": ("sala" in areas and "cucina" in areas),
                    "date_str":    ymd,
                })

    rows.sort(key=lambda r: (r["created_at"] or datetime.min), reverse=True)
    return rows

def local_day_range_utc(ymd: str):
    """
    Ritorna (start_utc, end_utc) per il giorno 'YYYY-MM-DD' in fuso Europe/Rome,
    convertiti in UTC (naive) per filtri DB.
    """
    day = datetime.strptime(ymd, "%Y-%m-%d").date()

    if TZ_ROME is not None:
        start_local = datetime(day.year, day.month, day.day, 0, 0, 0, tzinfo=TZ_ROME)
        end_local   = start_local + timedelta(days=1)
        start_utc = start_local.astimezone(timezone.utc).replace(tzinfo=None)
        end_utc   = end_local.astimezone(timezone.utc).replace(tzinfo=None)
    else:
        # fallback: considera “locale == UTC”
        start_utc = datetime(day.year, day.month, day.day, 0, 0, 0)
        end_utc   = start_utc + timedelta(days=1)

    return start_utc, end_utc


def parse_qty(s: str) -> Decimal:
    s = (s or "").strip().replace(",", ".")
    try:
        q = Decimal(s)
    except InvalidOperation:
        q = Decimal("0")
    if q < 0:
        q = Decimal("0")
    # arrotonda a 2 decimali
    return q.quantize(Decimal("0.01"))


def _filter_by_local_date(q, field, ymd: str):
    """
    Applica un filtro per data locale (Europe/Rome) in modo cross-DB.
    ymd deve essere 'YYYY-MM-DD'.
    """
    eng = (db.engine.name or "").lower()
    if eng.startswith("sqlite"):
        # SQLite: usa 'localtime' per convertire la stringa datetime in ora locale
        return q.filter(func.strftime('%Y-%m-%d', field, 'localtime') == ymd)
    elif 'postgresql' in eng:
        # Postgres: converti il timestamp in Europe/Rome e confronta su YYYY-MM-DD
        return q.filter(func.to_char(func.timezone('Europe/Rome', field), 'YYYY-MM-DD') == ymd)
    else:
        # Fallback: confronto semplice (potrebbe essere UTC)
        return q.filter(func.date(field) == ymd)


def combined_pairs_for_client_date(client_id: int, date_str: str):
    """
    Restituisce una lista di 'pair' ordinate per tempo per quel negozio in quel giorno (Europe/Rome).
    Ogni pair contiene, per ciascuna macro-area presente, {request_id, created_at, submitter_id}.
    """
    start_utc, end_utc = local_day_range_utc(date_str)

    reqs = (
        RequestHeader.query
        .filter_by(client_id=client_id, status="inviata")
        .filter(RequestHeader.created_at >= start_utc)
        .filter(RequestHeader.created_at <  end_utc)
        .options(joinedload(RequestHeader.items).joinedload(RequestItem.product).joinedload(Product.department))
        .order_by(RequestHeader.created_at.asc())
        .all()
    )

    # area -> [ {request_id, created_at, submitter_id} ... ] (ordine cronologico)
    area_lists = {}
    for rh in reqs:
        areas_for_rh = set()
        for it in rh.items:
            if (it.qty_requested or 0) > 0 and it.product and it.product.department:
                areas_for_rh.add((it.product.department.macro_area or "sala").lower())
        if not areas_for_rh:
            areas_for_rh = {"sala"}  # fallback difensivo

        for area in sorted(areas_for_rh):
            area_lists.setdefault(area, []).append({
                "request_id": rh.id,
                "created_at": rh.created_at,
                "submitter_id": getattr(rh, "submitted_by_user_id", None),
            })

    if not area_lists:
        return []

    # Pairing per indice tra TUTTE le aree
    n = max(len(lst) for lst in area_lists.values())
    pairs = []
    for i in range(n):
        areas = {}
        for area, lst in area_lists.items():
            if i < len(lst):
                areas[area] = lst[i]
        if not areas:
            continue
        created_at = max(block["created_at"] for block in areas.values() if block.get("created_at"))
        pairs.append({
            "rank": i + 1,
            "created_at": created_at,
            "areas": areas,                 # area -> block
        })

    pairs.sort(key=lambda p: p["created_at"], reverse=True)
    return pairs


def combined_pairs_for_client_date_any_areas(client_id: int, date_str: str):
    """
    Come la dashboard, ma supporta N macro-aree:
    - prende tutte le richieste 'inviata' del negozio nel giorno locale
    - per ogni macro-area crea la lista (ordinata) delle richieste
    - pairing per indice: 1a di ogni area insieme, poi 2a, ecc.
    - completezza: unione delle macro-aree abilitate degli 'invianti' presenti nel bundle
      (se mancano dati, fallback = le aree presenti nel bundle).
    Ritorna: [{rank, created_at, areas: {area: {request_id, created_at}}, is_complete, expected_areas}]
    """
    from collections import defaultdict

    start_utc, end_utc = local_day_range_utc(date_str)

    reqs = (
        RequestHeader.query
        .filter_by(client_id=client_id, status="inviata")
        .filter(RequestHeader.created_at >= start_utc)
        .filter(RequestHeader.created_at <  end_utc)
        .order_by(RequestHeader.created_at.asc())
        .all()
    )

    # macro-area -> lista ordinata
    area_lists = defaultdict(list)
    for rh in reqs:
        areas_in_rh = set()
        for it in rh.items:
            if (it.qty_requested or 0) <= 0: 
                continue
            if not it.product or not it.product.department:
                continue
            a = (it.product.department.macro_area or "sala").lower()
            areas_in_rh.add(a)

        # se non si riesce a dedurre un'area, fallback "sala"
        if not areas_in_rh:
            areas_in_rh.add("sala")

        for area in areas_in_rh:
            area_lists[area].append({
                "request_id": rh.id,
                "created_at": rh.created_at,
                "submitted_by": getattr(rh, "submitted_by", None),
            })

    # ordina ciascuna lista per tempo asc
    for lst in area_lists.values():
        lst.sort(key=lambda x: x["created_at"])

    n = max((len(lst) for lst in area_lists.values()), default=0)
    pairs = []
    for i in range(n):
        areas = {}
        submitters = []
        for area, lst in area_lists.items():
            if i < len(lst):
                rec = lst[i]
                areas[area] = {
                    "request_id": rec["request_id"],
                    "created_at": rec["created_at"],
                }
                if rec.get("submitted_by"):
                    submitters.append(rec["submitted_by"])

        if not areas:
            continue

        created_at = max(v["created_at"] for v in areas.values())

        # expected = unione delle macro-aree abilitate degli invianti presenti nel bundle
        expected = set()
        for u in submitters:
            expected |= set(allowed_macro_areas_for(u))  # dinamico per employee
        if not expected:
            # se non sappiamo chi ha inviato, non penalizziamo: atteso = aree presenti
            expected = set(areas.keys())

        is_complete = expected.issubset(set(areas.keys()))

        pairs.append({
            "rank": i + 1,
            "created_at": created_at,
            "areas": areas,
            "is_complete": is_complete,
            "expected_areas": sorted(expected),
        })

    pairs.sort(key=lambda p: p["created_at"], reverse=True)
    return pairs

def _items_for_request_in_area(req_id: int, expected_area: str):
    """Ritorna le righe della richiesta `req_id` che appartengono a `expected_area` ('sala'|'cucina')."""
    if not req_id:
        return []
    rh = RequestHeader.query.get(req_id)
    if not rh:
        return []

    rows = []
    for it in rh.items:
        if (it.qty_requested or 0) <= 0:
            continue
        if not it.product or not it.product.department:
            continue
        area = (it.product.department.macro_area or "sala")
        if area != expected_area:
            continue
        rows.append(it)

    # ordinamento leggibile
    rows.sort(
        key=lambda x: (
            x.product.department.name.lower() if x.product and x.product.department else "",
            x.product.name.lower() if x.product else ""
        )
    )
    return rows


def is_full_admin(u: User) -> bool:
    return u and u.role == "admin" and (u.admin_level or "full") == "full"

def build_plan_hierarchy_by_client(plan_id: int):
    """
    Ritorna una lista ordinata per client, poi per reparto, con soli qty_in > 0:
    [
      {
        'client': <User>,
        'departments': [
          {'name': 'Drogheria', 'products': [{'product': <Product>, 'qty_in': 3}, ...]},
          ...
        ]
      },
      ...
    ]
    """
    lines = (
    DistributionLine.query
        .filter(DistributionLine.plan_id == plan_id)
        .filter(DistributionLine.qty_in > 0)  # solo "Da Caricare"
        .options(
            joinedload(DistributionLine.product).joinedload(Product.department),
            joinedload(DistributionLine.client),
        )
        .all()
    )

    by_client = {}
    for ln in lines:
        client = ln.client
        dept_name = ln.product.department.name if ln.product and ln.product.department else "Senza reparto"
        slot = by_client.setdefault(client.id, {
            "client": client,
            "departments": defaultdict(list)
        })
        slot["departments"][dept_name].append({"product": ln.product, "qty_in": ln.qty_in})

    out = []
    for _, entry in by_client.items():
        depts = []
        for dname, rows in entry["departments"].items():
            rows_sorted = sorted(rows, key=lambda r: (r["product"].name or "").lower())
            depts.append({"name": dname, "products": rows_sorted})
        depts.sort(key=lambda d: d["name"].lower())
        out.append({"client": entry["client"], "departments": depts})

    out.sort(key=lambda b: (b["client"].display_name or "").lower())
    return out
    

# --- UTIL: slug sicuro per nomi file ---
import re
def _slug(s: str) -> str:
    if not s:
        return "senza_fornitore"
    return re.sub(r'[^A-Za-z0-9._-]+', '_', s).strip('_').lower() or "senza_fornitore"


def build_plan_hierarchy_by_supplier(plan_id: int):
    """
    Ritorna blocchi per fornitore:
    [
      {
        "supplier_name": str,
        "clients": [
           {"client": User, "departments":[
               {"name": dep_name, "products":[ {"product": Product, "qty_in": int} ]}
           ]}
        ]
      }, ...
    ]
    """
    client_blocks = build_plan_hierarchy_by_client(plan_id)  # già definita sopra

    def _supplier_name(prod):
        # 1) relazione Product.supplier.name (se esiste)
        sup_obj = getattr(prod, "supplier", None)
        if sup_obj and getattr(sup_obj, "name", None):
            return sup_obj.name
        # 2) campo libero product.supplier_name (se l'hai messo)
        name = getattr(prod, "supplier_name", None)
        if name:
            return name
        # 3) fallback “furbo”: usa il reparto come proxy del fornitore
        dep = getattr(prod, "department", None)
        if dep and getattr(dep, "name", None):
            return dep.name
        return "Senza fornitore"

    by_sup = {}  # supplier_name -> {"supplier_name": str, "clients":[...]}
    for cb in client_blocks:
        cl = cb["client"]
        for dep in cb.get("departments", []):
            for row in dep.get("products", []):
                prod = row["product"]
                qty  = int(row.get("qty_in") or 0)
                if qty <= 0:
                    continue

                sname = _supplier_name(prod)
                sup = by_sup.setdefault(sname, {"supplier_name": sname, "clients": []})

                # trova/crea client
                client_entry = next((x for x in sup["clients"] if x["client"].id == cl.id), None)
                if not client_entry:
                    client_entry = {"client": cl, "departments": []}
                    sup["clients"].append(client_entry)

                # trova/crea dept
                dep_entry = next((x for x in client_entry["departments"] if x["name"] == dep["name"]), None)
                if not dep_entry:
                    dep_entry = {"name": dep["name"], "products": []}
                    client_entry["departments"].append(dep_entry)

                dep_entry["products"].append({"product": prod, "qty_in": qty})

    # ordinamenti
    blocks = list(by_sup.values())
    blocks.sort(key=lambda b: b["supplier_name"].lower())
    for blk in blocks:
        blk["clients"].sort(key=lambda c: c["client"].display_name.lower())
        for d in blk["clients"]:
            d["departments"].sort(key=lambda x: x["name"].lower())
            for r in d["departments"]:
                r["products"].sort(key=lambda pr: pr["product"].name.lower())
    return blocks


import zipfile



# --- IMPORT necessari (in alto) ---
# aggiungi se non ci sono già:

# --- HELPER: macro-aree abilitate per l'utente ---
def allowed_macro_areas_for(user):
    """
    Restituisce la lista (ordinata) di macro-aree che l'utente può vedere.
    1) Se esistono righe in UserMacroAccess -> usa quelle.
    2) Altrimenti fallback ai vecchi flag (sala/cucina).
    3) Come ultima risorsa (nessun dato), non concedere nulla.
    """
    rows = UserMacroAccess.query.filter_by(user_id=user.id).all()
    areas = sorted({r.area for r in rows if r.area})

    if areas:
        return areas

    # Fallback legacy
    legacy = []
    if getattr(user, "access_sala", False):
        legacy.append("sala")
    if getattr(user, "access_cucina", False):
        legacy.append("cucina")

    return sorted(set(legacy))

@app.get("/admin/griglia/plan/<int:plan_id>/download-suppliers")
@login_required
@require_role("admin")
def admin_grid_plan_download_suppliers(plan_id):
    plan = DistributionPlan.query.get_or_404(plan_id)
    blocks = build_plan_hierarchy_by_supplier(plan_id)

    if not blocks:
        flash("Nessun prodotto da caricare per questo piano.", "warning")
        return redirect(url_for("admin_grid", plan_id=plan.id))

    from weasyprint import HTML, CSS

    # CSS (stessi della stampa singola)
    css_files = []
    css_main_path = os.path.join(BASE_DIR, "static", "css", "style.css")
    if os.path.exists(css_main_path):
        css_files.append(CSS(filename=css_main_path))
    css_files.append(CSS(string="@page { size: A4; margin: 16mm; }"))

    # ZIP in-memory
    zip_buffer = io.BytesIO()
    used_names = set()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for blk in blocks:
            supplier_name = blk["supplier_name"]

            html = render_template(
                "admin/plan_print_supplier.html",
                plan=plan,
                clients=blk["clients"],
                gen_dt=datetime.utcnow(),
            )
            pdf_bytes = HTML(string=html, base_url=BASE_DIR).write_pdf(stylesheets=css_files)

            # solo nome fornitore (sanificato) come nome file
            base = _slug(supplier_name) or "fornitore"
            name = f"{base}.pdf"
            i = 2
            while name in used_names:
                name = f"{base}-{i}.pdf"
                i += 1
            used_names.add(name)

            zf.writestr(name, pdf_bytes)
    zip_buffer.seek(0)
    return send_file(   
        zip_buffer,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"recap-piano-{plan.id}-per-fornitore.zip",
        max_age=0,
        conditional=False,
        etag=False,
        last_modified=None,
    )


@app.get("/admin/griglia/plan/<int:plan_id>/download")
@login_required
@require_role("admin")
def admin_grid_plan_download(plan_id):
    # 1) Costruisco i dati con lo stesso template HTML che usi a schermo
    plan = DistributionPlan.query.get_or_404(plan_id)
    hierarchy = build_plan_hierarchy_by_client(plan_id)

    html = render_template(
        "admin/plan_print.html",   # <-- lo stesso template usato prima
        plan=plan,
        hierarchy=hierarchy,
        gen_dt=datetime.utcnow(),
    )

    # 2) Converto in PDF con WeasyPrint e lo SCARICO (attachment)
    from weasyprint import HTML, CSS

    css_files = []
    # Iniettiamo anche il tuo CSS principale per mantenere lo stile
    css_main_path = os.path.join(BASE_DIR, "static", "css", "style.css")
    if os.path.exists(css_main_path):
        css_files.append(CSS(filename=css_main_path))

    # Margini pagina (A4) + eventuale tuning di stampa
    css_files.append(CSS(string="""
        @page { size: A4; margin: 16mm; }
        /* Evita divider troppo scuri su stampa, se servisse
        .table td, .table th { border-color: #ddd !important; } */
    """))

    # base_url IMPORTANTISSIMO per risolvere url relativi/icone/font
    pdf_bytes = HTML(string=html, base_url=BASE_DIR).write_pdf(stylesheets=css_files)

    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype="application/pdf",
        as_attachment=True,  # <-- forza download
        download_name=f"recap-piano-{plan.id}.pdf",
        max_age=0,
        conditional=False,
        etag=False,
        last_modified=None,
    )

# ---------- ROUTES COMUNI ----------

@app.route("/")
@login_required
def index():
    if current_user.role == "admin":
        return redirect(url_for("admin_dashboard"))
    if current_user.role == "employee":
        return redirect(url_for("client_departments"))
    # i 'client' (negozi) non devono più loggarsi
    flash("Profilo non abilitato all’accesso.", "warning")
    return redirect(url_for("logout"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user = User.query.filter_by(username=username).first()

        # Solo admin o employee possono autenticarsi
        if user and user.role in ("admin", "employee") and user.check_password(password):
            login_user(user)
            session.pop("macro_unlock", None)  # reset sblocchi area
            flash("Accesso effettuato!", "success")
            return redirect(url_for("index"))
        else:
            flash("Credenziali non valide", "danger")
    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    # Se un employee ha una bozza con righe >0 e NON ha confermato, chiedi conferma server-side
    if current_user.role == "employee":
        pending_rows = _draft_filled_rows_count_for_current_employee()
        confirmed = request.args.get("confirm") == "1"
        if pending_rows > 0 and not confirmed:
            # Mostra pagina di conferma dedicata
            return render_template(
                "confirm_logout.html",
                pending_rows=pending_rows
            )

    logout_user()
    session.pop("macro_unlock", None)
    flash("Disconnesso.", "info")
    return redirect(url_for("login"))

# ---------- ROUTES CLIENT ----------
@app.route("/client/reparti")
@login_required
@require_roles("client", "employee")
def client_departments():
    # Reparti attivi
    departments = (
        Department.query
        .filter_by(active=True)
        .order_by(Department.name)
        .all()
    )

    # Bozza per il negozio effettivo
    draft = get_or_create_draft_request(current_shop_id())

    # Conteggio righe compilate per reparto
    counts = {}
    for it in draft.items:
        if it.product and it.product.department_id and (it.qty_requested or 0) > 0:
            counts[it.product.department_id] = counts.get(it.product.department_id, 0) + 1

    # Aree note e aree permesse all'utente
    all_areas = list_macro_areas()  # es. ["sala","cucina","prova", ...]
    if current_user.role == "employee":
        allowed = set(allowed_macro_areas_for(current_user))
    else:
        allowed = set(all_areas)

    # Reparti per area (solo permesse)
    deps_by_area = {}
    for d in departments:
        area = (d.macro_area or "sala").lower()
        if area in allowed:
            deps_by_area.setdefault(area, []).append(d)

    # Info sblocco per ogni area (mostrale SEMPRE se permesse)
    unlocked = session.get("macro_unlock", {})  # {"sala": True, ...}
    areas = []
    for area in all_areas:
        if area not in allowed:
            continue
        code = get_macro_code(area) or ""
        deps = sorted(deps_by_area.get(area, []), key=lambda x: (x.name or "").lower())
        areas.append({
            "area": area,
            "deps": deps,                          # può essere lista vuota
            "code_required": (code != ""),
            "unlocked": unlocked.get(area, False) or (code == ""),
        })

    # Totale righe >0 per abilitare invio
    total_selected = sum(1 for it in draft.items if (it.qty_requested or 0) > 0)

    return render_template(
        "client/departments.html",
        areas=areas,
        counts=counts,
        total_selected=total_selected,
    )


@app.route("/client/reparti/<int:dep_id>", methods=["GET", "POST"])
@login_required
@require_roles("employee")
def client_products(dep_id):
    department = Department.query.get_or_404(dep_id)

    # Permesso: macro-area del reparto deve essere nelle aree abilitate
    area = (department.macro_area or "sala").lower()
    allowed = allowed_macro_areas_for(current_user)
    if area not in allowed:
        flash("Non hai accesso a questa macro-area.", "danger")
        return redirect(url_for("client_departments", area=allowed[0] if allowed else None))

    # Codice area (se richiesto)
    code_required = (get_macro_code(area) != "")
    unlocked = session.get("macro_unlock", {})
    if code_required and not unlocked.get(area, False):
        flash(f"Area '{area}' protetta: inserisci il codice per accedere.", "warning")
        return redirect(url_for("client_departments"))

    q = request.args.get("q", "").strip()

    base_q = Product.query.filter_by(department_id=dep_id, active=True)
    if q:
        base_q = base_q.filter(func.lower(Product.name).like(f"%{q.lower()}%"))
    products = base_q.order_by(Product.name).all()

    client_id_for_work = current_user.works_for_client_id
    draft = get_or_create_draft_request(client_id_for_work)

    if request.method == "POST":
        for p in products:
            from decimal import Decimal  # in alto se serve

            qty_str = (request.form.get(f"qty_{p.id}") or "0").strip()
            qty = parse_qty(qty_str)          # -> Decimal con 2 decimali, >= 0
            note = (request.form.get(f"note_{p.id}") or "").strip() or None

            item = next((i for i in draft.items if i.product_id == p.id), None)
            if item is None:
                item = RequestItem(request_id=draft.id, product_id=p.id,
                                qty_requested=qty, note=note)
                db.session.add(item)
            else:
                item.qty_requested = qty
                item.note = note
        db.session.commit()
        flash("Quantità salvate.", "success")
        return redirect(url_for("client_departments"))

    qty_map = {i.product_id: i for i in draft.items}
    return render_template("client/products.html", department=department, products=products, qty_map=qty_map, q=q)



@app.route("/client/riepilogo", methods=["GET", "POST"])
@login_required
@require_roles("employee")
def client_review():
    client_id_for_work = current_user.works_for_client_id
    draft = get_or_create_draft_request(client_id_for_work)
    draft = RequestHeader.query.get(draft.id)  # ricarica per sicurezza

    # Solo righe con quantità > 0
    items = [i for i in draft.items if (i.qty_requested or 0) > 0]

    from collections import defaultdict

    def dep_name_of(it):
        if it.product and it.product.department and it.product.department.name:
            return it.product.department.name
        return "Senza reparto"

    # Aree consentite per questo employee
    allowed_areas = set(allowed_macro_areas_for(current_user))

    # Raggruppo DINAMICAMENTE: macro_area -> reparto -> [items]
    grouped = defaultdict(lambda: defaultdict(list))
    for it in items:
        area = (getattr(getattr(it.product, "department", None), "macro_area", "sala") or "sala").lower()
        if allowed_areas and area not in allowed_areas:
            continue
        grouped[area][dep_name_of(it)].append(it)

    # Ordina reparti e prodotti
    def sort_blocks(area_map: dict):
        blocks = []
        for dep, rows in area_map.items():
            rows_sorted = sorted(rows, key=lambda x: (x.product.name or "").lower() if x.product else "")
            blocks.append({"department": dep, "rows": rows_sorted})
        blocks.sort(key=lambda b: (b["department"] or "").lower())
        return blocks

    # Ordine aree: quello definito in amministrazione; poi eventuali aree “extra” incontrate
    areas_order = list_macro_areas()  # es. ["sala","cucina","detersivi", ...]
    grouped_macro_sorted = {}
    for a in areas_order:
        if a in grouped:
            grouped_macro_sorted[a] = sort_blocks(grouped[a])
    for a in grouped.keys():
        if a not in grouped_macro_sorted:
            grouped_macro_sorted[a] = sort_blocks(grouped[a])

    if request.method == "POST":
        if len(items) == 0:
            flash("Aggiungi almeno una quantità prima di inviare.", "warning")
            return redirect(url_for("client_departments"))

        draft.created_at = datetime.utcnow()
        draft.status = "inviata"
        # 👇 salva chi l'ha inviata
        draft.submitted_by_user_id = current_user.id

        db.session.commit()

        flash("Giacenza inviata al magazzino!", "success")
        return redirect(url_for("client_departments"))

    # GET → mostra pagina
    return render_template(
        "client/review.html",
        draft=draft,
        items=items,
        grouped_macro=grouped_macro_sorted
    )




def _draft_filled_rows_count_for_current_employee() -> int:
    """
    Ritorna quante righe >0 ha la bozza del negozio dell'employee loggato.
    0 se non ci sono righe oppure se non è un employee.
    """
    try:
        if not current_user.is_authenticated or current_user.role != "employee":
            return 0
        client_id = current_user.works_for_client_id
        draft = (
            RequestHeader.query
            .filter_by(client_id=client_id, status="bozza")
            .order_by(RequestHeader.created_at.desc())
            .first()
        )
        if not draft:
            return 0
        return sum(1 for it in draft.items if (it.qty_requested or 0) > 0)
    except Exception:
        return 0


# ---------- ROUTES ADMIN ----------
@app.route("/admin")
@login_required
@require_role("admin")
def admin_dashboard():
    # data dal querystring, default oggi (UTC)
    date_str = request.args.get("date", "").strip()
    if date_str:
        try:
            selected_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            selected_date = (datetime.now(TZ_ROME).date() if TZ_ROME else datetime.utcnow().date())
    else:
        selected_date = (datetime.now(TZ_ROME).date() if TZ_ROME else datetime.utcnow().date())
    date_filter = selected_date.strftime("%Y-%m-%d")

    # Costruisco righe: 1 card per ogni "pair" di quel giorno
    # Costruisco righe: 1 card per ogni "pair" di quel giorno
    shops = User.query.filter_by(role="client").order_by(User.display_name).all()
    rows = []  # {"client":User, "rank":int, "created_at":datetime, "is_complete":bool}
    for c in shops:
        pairs = combined_pairs_for_client_date(c.id, date_filter)
        for p in pairs:

            present_areas = set((p.get("areas") or {}).keys())

            # Unione delle aree abilitate dei mittenti (se tracciati)
            required_areas = set()
            for area, block in (p.get("areas") or {}).items():
                submitter_id = block.get("submitter_id")
                if submitter_id:
                    u = User.query.get(submitter_id)
                    if u:
                        required_areas |= set(allowed_macro_areas_for(u))

            # Fallback (storico senza submitter): richiedi solo ciò che è presente
            if not required_areas:
                required_areas = set(present_areas)
            # completo solo se ho sia sala che cucina nel pair
            is_complete = required_areas.issubset(present_areas)

            rows.append({
                "client_id": c.id,
                "client_name": c.display_name or c.username,
                "rank": p["rank"],
                "created_at": p["created_at"],
                "is_complete": is_complete,  # <-- boolean garantito
            })

    rows.sort(key=lambda r: (r["created_at"] or datetime.min), reverse=True)

    total_clients = User.query.filter_by(role="client").count()
    total_products = Product.query.count()
    total_departments = Department.query.count()

    return render_template(
        "admin/dashboard.html",
        rows=rows,
        total_clients=total_clients,
        total_products=total_products,
        total_departments=total_departments,
        date_filter=date_filter,
        giacenze_count=len(rows),
    )

@app.route("/admin/richieste/<int:req_id>", methods=["GET"], endpoint="admin_request_detail")
@login_required
@require_role("admin")
def admin_request_detail(req_id):
    rh = RequestHeader.query.get_or_404(req_id)
    # raggruppo per reparto (come prima)
    grouped = {}
    for item in rh.items:
        if (item.qty_requested or 0) <= 0:
            continue
        dep = item.product.department.name if item.product and item.product.department else "—"
        grouped.setdefault(dep, []).append(item)
    return render_template("admin/request_detail.html", rh=rh, grouped=grouped)

@app.post("/admin/richieste/pair/delete")
@login_required
@require_role("admin")
def admin_request_pair_delete():
    # Leggo gli ID (possono arrivarne 1 o 2)
    rid1 = request.form.get("req_id_1", type=int)
    rid2 = request.form.get("req_id_2", type=int)
    date = (request.form.get("date") or "").strip()

    req_ids = [rid for rid in (rid1, rid2) if rid]
    if not req_ids:
        flash("Nessuna giacenza da eliminare.", "warning")
        return redirect(url_for("admin_dashboard", date=date) if date else url_for("admin_dashboard"))

    # Recupero le RequestHeader e le elimino (items vengono rimossi via cascade)
    headers = RequestHeader.query.filter(RequestHeader.id.in_(req_ids)).all()
    if not headers:
        flash("Giacenze non trovate.", "danger")
        return redirect(url_for("admin_dashboard", date=date) if date else url_for("admin_dashboard"))

    for h in headers:
        db.session.delete(h)
    db.session.commit()

    flash(f"Eliminata coppia di giacenze ({len(headers)} record).", "success")
    return redirect(url_for("admin_dashboard", date=date) if date else url_for("admin_dashboard"))

@app.post("/admin/richieste/<int:req_id>/delete")
@login_required
@require_role("admin")
def admin_request_delete(req_id):
    rh = RequestHeader.query.get_or_404(req_id)

    # (opzionale) consenti solo se è "inviata"
    # if rh.status != "inviata":
    #     flash("Puoi eliminare solo giacenze 'inviata'.", "warning")
    #     return redirect(url_for("admin_request_detail", req_id=req_id))

    db.session.delete(rh)
    db.session.commit()
    flash("Giacenza eliminata.", "success")
    # Torna alla lista storica o alla dashboard, come preferisci
    return redirect(request.referrer or url_for("admin_requests"))

@app.route("/admin/cliente/<int:client_id>/giorno/<date_str>/pair/<int:rank>")
@login_required
@require_role("admin")
def admin_client_pair_detail(client_id, date_str, rank):
    # parse data
    try:
        selected_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return redirect(url_for("admin_dashboard"))

    date_filter = selected_date.strftime("%Y-%m-%d")

    # Trova la pair (serve per link di ritorno e per orari sala/cucina in testata)
    pairs = combined_pairs_for_client_date(client_id, date_filter)
    pair = next((p for p in pairs if p.get("rank") == rank), None)
    if not pair:
        flash("Coppia non trovata per il negozio e la data selezionati.", "warning")
        return redirect(url_for("admin_dashboard", date=date_filter))

    # Estraggo eventuali requestId sala/cucina per mostrare gli orari in testata
    areas_pair = pair.get("areas") or {}
    sala_req_id   = (areas_pair.get("sala") or {}).get("request_id")
    cucina_req_id = (areas_pair.get("cucina") or {}).get("request_id")
    sala_request   = RequestHeader.query.get(sala_req_id)   if sala_req_id   else None
    cucina_request = RequestHeader.query.get(cucina_req_id) if cucina_req_id else None

    # Prendo tutte le richieste INVIATE di quel giorno (locale Europe/Rome) per quel negozio
    start_utc, end_utc = local_day_range_utc(date_filter)
    day_requests = (
        RequestHeader.query
        .filter_by(client_id=client_id, status="inviata")
        .filter(RequestHeader.created_at >= start_utc)
        .filter(RequestHeader.created_at <  end_utc)
        .options(
            joinedload(RequestHeader.items)
              .joinedload(RequestItem.product)
              .joinedload(Product.department)
        )
        .order_by(RequestHeader.created_at.asc())
        .all()
    )

    # Raggruppo DINAMICAMENTE: area -> reparto -> [items]
    from collections import defaultdict
    areas_map = defaultdict(lambda: defaultdict(list))

    for rh in day_requests:
        for it in rh.items:
            qty = it.qty_requested or 0
            if qty <= 0:
                continue
            if not it.product or not it.product.department:
                continue
            area = (it.product.department.macro_area or "sala").lower()
            dep_name = it.product.department.name or "Senza reparto"
            areas_map[area][dep_name].append(it)

    # Ordino reparti e prodotti
    def sort_blocks(area_map: dict):
        blocks = []
        for dep, rows in area_map.items():
            rows_sorted = sorted(rows, key=lambda x: (x.product.name or "").lower())
            blocks.append({"department": dep, "rows": rows_sorted})
        blocks.sort(key=lambda b: (b["department"] or "").lower())
        return blocks

    # Ordine aree = quello configurato + eventuali aree non in elenco
    ordered = {}
    for a in list_macro_areas():            # es. ["sala","cucina","detersivi", ...]
        if a in areas_map:
            ordered[a] = sort_blocks(areas_map[a])
    for a in areas_map.keys():
        if a not in ordered:
            ordered[a] = sort_blocks(areas_map[a])

    client = User.query.get_or_404(client_id)

    return render_template(
        "admin/client_pair_detail.html",
        client=client,
        date_filter=date_filter,
        rank=rank,
        sala_request=sala_request,       # solo per timestamp in testata (se esiste)
        cucina_request=cucina_request,   # idem
        areas_blocks=ordered             # <<--- NUOVO: tutte le aree con blocchi/righe
    )

@app.route("/admin/clients", methods=["GET", "POST"])
@login_required
@require_role("admin")
def admin_clients():
    if request.method == "POST":
        if not is_full_admin(current_user):
            flash("Permessi insufficienti: non puoi creare/modificare i negozi.", "danger")
            return redirect(url_for("admin_clients"))

        # Arriva solo il nome punto vendita dal form
        display_name = (request.form.get("display_name") or "").strip()
        if not display_name:
            flash("Nome punto vendita richiesto.", "danger")
            return redirect(url_for("admin_clients"))

        # Se manca username, generane uno dal display_name
        raw_username = (request.form.get("username") or "").strip()
        if not raw_username:
            base = _slug(display_name) or "negozio"
            username = base
            i = 2
            while User.query.filter_by(username=username).first():
                suffix = f"-{i}"
                username = (base[:80 - len(suffix)] + suffix)
                i += 1
        else:
            username = raw_username

        # (Se hai passato uno username manuale, garantisci l'unicità)
        if User.query.filter_by(username=username).first():
            flash("Esiste già un negozio con questo nome. Riprova con un nome diverso.", "danger")
            return redirect(url_for("admin_clients"))

        # Crea NEGOZIO (i client non fanno login: password random)
        u = User(username=username, display_name=display_name, role="client")
        u.set_password(os.urandom(16).hex())
        db.session.add(u)
        db.session.commit()
        flash("Negozio creato.", "success")
        return redirect(url_for("admin_clients"))

    # GET
    clients = User.query.filter_by(role="client").order_by(User.display_name).all()
    macro_areas = MacroAreaAccess.query.order_by(MacroAreaAccess.area.asc()).all()
    return render_template("admin/clients.html", clients=clients, macro_areas=macro_areas)


@app.post("/admin/macro-area/create")
@login_required
@require_role("admin")
def admin_macro_area_create():
    if is_partial_admin():
        flash("Permessi insufficienti.", "danger")
        return redirect(request.referrer or url_for("admin_users"))

    name = (request.form.get("name") or "").strip().lower()
    # normalizza
    import re
    name = re.sub(r"[^a-z0-9_-]+", "_", name).strip("_")
    if not name:
        flash("Nome non valido.", "danger")
        return redirect(request.referrer or url_for("admin_users"))

    if MacroAreaAccess.query.filter_by(area=name).first():
        flash("Macro-area già presente.", "warning")
        return redirect(request.referrer or url_for("admin_users"))

    db.session.add(MacroAreaAccess(area=name, code=""))
    db.session.commit()
    flash("Macro-area creata.", "success")
    return redirect(request.referrer or url_for("admin_users"))


@app.post("/admin/macro-area/<int:area_id>/update-code")
@login_required
@require_role("admin")
def admin_macro_area_update_code(area_id):
    a = MacroAreaAccess.query.get_or_404(area_id)

    # nuovo nome (solo rename: niente codici)
    new_name = (request.form.get("name") or "").strip().lower()
    if not new_name:
        flash("Il nome della macro-area è obbligatorio.", "danger")
        return redirect(request.referrer or url_for("admin_clients"))

    # unicità
    exists = (MacroAreaAccess.query
              .filter(MacroAreaAccess.area == new_name, MacroAreaAccess.id != a.id)
              .first())
    if exists:
        flash("Esiste già una macro-area con questo nome.", "danger")
        return redirect(request.referrer or url_for("admin_clients"))

    a.area = new_name
    db.session.commit()
    flash("Macro-area aggiornata.", "success")
    return redirect(request.referrer or url_for("admin_clients"))


@app.post("/admin/macro-area/<int:area_id>/delete")
@login_required
@require_role("admin")
def admin_macro_area_delete(area_id):
    if is_partial_admin():
        flash("Permessi insufficienti.", "danger")
        return redirect(request.referrer or url_for("admin_users"))

    m = MacroAreaAccess.query.get_or_404(area_id)
    in_use = Department.query.filter_by(macro_area=m.area).count()
    if in_use:
        flash("Impossibile eliminare: ci sono reparti che usano questa macro-area.", "danger")
        return redirect(request.referrer or url_for("admin_users"))

    db.session.delete(m)
    db.session.commit()
    flash("Macro-area eliminata.", "success")
    return redirect(request.referrer or url_for("admin_users"))


@app.route("/admin/richieste/cliente/<int:client_id>/ultima")
@login_required
@require_role("admin")
def admin_client_last_combined(client_id):
    client = User.query.get_or_404(client_id)

    last = (
        RequestHeader.query
        .filter_by(client_id=client_id, status="inviata")
        .order_by(RequestHeader.created_at.desc())
        .first()
    )
    if not last:
        flash("Questo negozio non ha giacenze inviate.", "warning")
        return redirect(url_for("admin_clients"))

    date_str = to_rome(last.created_at).strftime("%Y-%m-%d")
    groups = combined_client_items_for_date(client_id, date_str)

    return render_template(
        "admin/client_last_request.html",
        client=client,
        date_filter=date_str,
        groups=groups,
    )

@app.route("/admin/utenti", methods=["GET", "POST"])
@login_required
@require_role("admin")
def admin_users():
    # Negozi (per associare gli employee)
    shops = User.query.filter_by(role="client").order_by(User.display_name).all()

    if request.method == "POST":
        if is_partial_admin():
            flash("Non autorizzato: questo account admin non può creare utenti.", "danger")
            return redirect(url_for("admin_users"))

        role     = (request.form.get("role") or "employee").strip().lower()
        username = (request.form.get("username") or "").strip()
        display  = (request.form.get("display_name") or "").strip() or username
        password = (request.form.get("password") or "").strip()

        if not (username and password):
            flash("Username e password sono obbligatori.", "danger")
            return redirect(url_for("admin_users"))

        if User.query.filter_by(username=username).first():
            flash("Username già esistente.", "danger")
            return redirect(url_for("admin_users"))

        # --- CREA ADMIN ---
        if role == "admin":
            admin_level = (request.form.get("admin_level") or "full").strip().lower()
            if admin_level not in ("full", "partial"):
                admin_level = "full"

            u = User(
                username=username,
                display_name=display,
                role="admin",
                admin_level=admin_level
            )
            u.set_password(password)
            db.session.add(u)
            db.session.commit()
            flash("Admin creato.", "success")
            return redirect(url_for("admin_users"))

        # --- CREA EMPLOYEE (utente negozio) con macro-aree dinamiche ---
        shop_id = request.form.get("shop_id", type=int)
        store = User.query.filter_by(id=shop_id, role="client").first()
        if not store:
            flash("Negozio non valido.", "danger")
            return redirect(url_for("admin_users"))

        # macro-aree selezionate dal form
        selected_areas = [a.strip().lower() for a in request.form.getlist("areas[]")]

        # valida contro le macro-aree esistenti
        valid_areas = {m.area for m in MacroAreaAccess.query.all()}
        selected_areas = [a for a in selected_areas if a in valid_areas]

        emp = User(
            username=username,
            display_name=display,
            role="employee",
            works_for_client_id=shop_id,
            # fallback legacy per parti dell'app che leggono ancora questi flag
            access_sala=("sala" in selected_areas),
            access_cucina=("cucina" in selected_areas),
        )
        emp.set_password(password)
        db.session.add(emp)
        db.session.flush()  # per avere emp.id

        # salva permessi macro-area dinamici
        # (nessun record pregresso da cancellare perché l'utente è nuovo)
        for area in selected_areas:
            db.session.add(UserMacroAccess(user_id=emp.id, area=area))

        db.session.commit()
        flash("Utente negozio creato.", "success")
        return redirect(url_for("admin_users"))

    # --- GET: elenco utenti + macro-aree per la UI ---
    users = (
        User.query
        .filter(User.role.in_(["admin", "employee"]))
        .order_by(User.role.desc(), User.display_name.asc())
        .all()
    )

    macro_areas = MacroAreaAccess.query.order_by(MacroAreaAccess.area.asc()).all()

    links = UserMacroAccess.query.all()
    user_area_map = {}
    for link in links:
        user_area_map.setdefault(link.user_id, []).append(link.area)


    return render_template(
        "admin/employees.html",
        shops=shops,
        users=users,
        macro_areas=macro_areas,      # <-- per i checkbox dinamici nel form
        user_area_map=user_area_map,  # <-- per mostrare le aree assegnate in tabella
    )
@app.post("/admin/utenti/<int:user_id>/update")
@login_required
@require_role("admin")
def admin_users_update(user_id):
    u = User.query.get_or_404(user_id)

    # EMPLOYEE: aggiorna negozio + accessi macro-area + (eventuale) password
    if u.role == "employee":
        display  = (request.form.get("display_name") or "").strip() or u.username
        password = (request.form.get("password") or "").strip()
        shop_id  = request.form.get("shop_id", type=int)

        store = User.query.filter_by(id=shop_id, role="client").first()
        if not store:
            flash("Negozio non valido.", "danger")
            return redirect(request.referrer or url_for("admin_departments"))

        # macro-aree selezionate (dinamico)
        selected_areas = [a.strip().lower() for a in request.form.getlist("areas[]")]
        valid_areas = {m.area for m in MacroAreaAccess.query.all()}
        selected_areas = [a for a in selected_areas if a in valid_areas]

        # aggiorna anagrafica
        u.display_name = display
        u.works_for_client_id = shop_id
        # legacy flags per compatibilità con parti vecchie dell’UI
        u.access_sala   = ("sala"   in selected_areas)
        u.access_cucina = ("cucina" in selected_areas)
        if password:
            u.set_password(password)
        db.session.commit()

        # aggiorna pivot user <-> macro-aree
        UserMacroAccess.query.filter_by(user_id=u.id).delete(synchronize_session=False)
        for area in selected_areas:
            db.session.add(UserMacroAccess(user_id=u.id, area=area))
        db.session.commit()

        flash("Utente negozio aggiornato.", "success")
        return redirect(request.referrer or url_for("admin_departments"))

    # ADMIN invariato (lascia il tuo codice per gli admin)
    if u.role == "admin":
        display  = (request.form.get("display_name") or "").strip() or u.username
        password = (request.form.get("password") or "").strip()
        admin_level = (request.form.get("admin_level") or u.admin_level or "full").strip().lower()
        if admin_level not in ("full", "partial"):
            admin_level = "full"

        u.display_name = display
        u.admin_level = admin_level
        if password:
            u.set_password(password)

        db.session.commit()
        flash("Admin aggiornato.", "success")
        return redirect(request.referrer or url_for("admin_departments"))

    flash("Tipo utente non gestito per l'aggiornamento.", "danger")
    return redirect(request.referrer or url_for("admin_departments"))


@app.post("/admin/utenti/<int:user_id>/delete")
@login_required
@require_role("admin")
def admin_users_delete(user_id):
    if is_partial_admin():
        flash("Non autorizzato: questo account admin non può eliminare utenti.", "danger")
        return redirect(request.referrer or url_for("admin_departments"))
    u = User.query.get_or_404(user_id)
    if u.role not in ("employee","admin"):
        flash("Tipo utente non eliminabile qui.", "danger")
        return redirect(request.referrer or url_for("admin_departments"))

    # (eventuali cleanup come già fai)
    db.session.delete(u)
    db.session.commit()
    flash("Account eliminato.", "success")
    return redirect(request.referrer or url_for("admin_departments"))

@app.route("/admin/clients/<int:client_id>")
@login_required
@require_role("admin")
def admin_client_detail(client_id):
    client = User.query.get_or_404(client_id)
    # Mostra richieste inviate dal più recente
    requests = RequestHeader.query.filter_by(client_id=client.id, status="inviata").order_by(RequestHeader.created_at.desc()).all()
    return render_template("admin/client_detail.html", client=client, requests=requests)


@app.route("/admin/richieste")
@login_required
@require_role("admin")
def admin_requests():
    # Filtri: giorno singolo (date) oppure range (from/to)
    date_str = (request.args.get("date") or "").strip()
    from_str = (request.args.get("from")  or "").strip()
    to_str   = (request.args.get("to")    or "").strip()

    # Prendo tutte le richieste (solo per ricavare i giorni locali per client)
    base_q = (
        RequestHeader.query
        .filter(RequestHeader.status == "inviata")
        .options(
            joinedload(RequestHeader.client),
            joinedload(RequestHeader.items).joinedload(RequestItem.product).joinedload(Product.department),
            joinedload(RequestHeader.submitted_by)
        )
    )

    if date_str:
        s, e = local_day_range_utc(date_str)
        base_q = base_q.filter(RequestHeader.created_at >= s, RequestHeader.created_at < e)
        from_filter, to_filter = "", ""
    else:
        from_filter = from_str
        to_filter   = to_str
        if from_str:
            s, _ = local_day_range_utc(from_str)
            base_q = base_q.filter(RequestHeader.created_at >= s)
        if to_str:
            _, e = local_day_range_utc(to_str)
            base_q = base_q.filter(RequestHeader.created_at < e)

    all_reqs = base_q.order_by(RequestHeader.created_at.desc()).all()

    # Giorni locali presenti per ogni client
    from collections import defaultdict
    ymd_by_client = defaultdict(set)
    for rh in all_reqs:
        ymd = _local_ymd(rh.created_at)
        ymd_by_client[rh.client_id].add(ymd)

    # Costruisco le "righe" (card) come la dashboard: una per (client, giorno, rank)
    rows = []  # {"client_id","client_name","rank","created_at","is_complete","date_str"}

    # ordino i client per nome per stabilità, ma le card verranno poi ordinate per created_at
    clients = {rh.client_id: rh.client for rh in all_reqs if rh.client_id}
    for client_id, ymd_set in ymd_by_client.items():
        client = clients.get(client_id)
        client_name = (client.display_name if client else f"#{client_id}")
        for ymd in sorted(ymd_set):  # puoi invertire se vuoi
            pairs = combined_pairs_for_client_date_any_areas(client_id, ymd)
            for p in pairs:
                rows.append({
                    "client_id": client_id,
                    "client_name": client_name,
                    "rank": p["rank"],
                    "created_at": p["created_at"],
                    "is_complete": p["is_complete"],
                    "date_str": ymd,
                })

    # Ordina le card per tempo (discendente) così vedi le più recenti in alto
    rows.sort(key=lambda r: (r["created_at"] or datetime.min), reverse=True)

    return render_template(
        "admin/requests.html",
        rows=rows,
        date_filter=date_str,
        from_filter=from_filter,
        to_filter=to_filter,
    )

@app.route("/admin/reparti", methods=["GET", "POST"])
@login_required
@require_role("admin")
def admin_departments():
    if request.method == "POST":
        form_type = request.form.get("form_type", "create_dep")

        # --- CREA REPARTO ---
        if form_type == "create_dep":
            if is_partial_admin():
                flash("Non autorizzato: questo account admin non può creare reparti.", "danger")
                return redirect(url_for("admin_departments"))

            name = (request.form.get("name") or "").strip()
            macro_area = (request.form.get("macro_area") or "").strip().lower()

            valid_areas = set(list_macro_areas())
            if macro_area not in valid_areas:
                flash("Macro-area non valida. Crea prima la macro-area.", "danger")
                return redirect(url_for("admin_departments"))

            if not name:
                flash("Nome richiesto.", "danger")
            elif Department.query.filter_by(name=name).first():
                flash("Reparto già esistente.", "danger")
            else:
                d = Department(name=name, active=True, macro_area=macro_area)
                db.session.add(d)
                db.session.commit()
                flash("Reparto creato.", "success")

            return redirect(url_for("admin_departments"))

        # --- CODICI MACRO-AREA (se ancora usati nella UI) ---
        elif form_type == "codes":
            code_sala   = (request.form.get("code_sala")   or "").strip()
            code_cucina = (request.form.get("code_cucina") or "").strip()
            set_macro_code("sala", code_sala)
            set_macro_code("cucina", code_cucina)
            flash("Codici aggiornati.", "success")
            return redirect(url_for("admin_departments"))

        # --- CREA UTENTE (admin / employee) ---
        elif form_type == "create_user":
            if is_partial_admin():
                flash("Non autorizzato: questo account admin non può creare utenti.", "danger")
                return redirect(url_for("admin_departments"))

            role        = (request.form.get("role") or "employee").strip().lower()
            username    = (request.form.get("username") or "").strip()
            password    = (request.form.get("password") or "").strip()
            admin_level = (request.form.get("admin_level") or "full").strip().lower()

            # campi employee
            shop_id       = request.form.get("shop_id", type=int)

            # NUOVO: macro-aree selezionate dal form (multi-selezione)
            areas_selected = request.form.getlist("areas[]")  # es: ["sala","cucina","ortofrutta", ...]
            # fallback legacy: se non inviato nulla, ricavo da check sala/cucina
            access_sala   = bool(request.form.get("access_sala"))
            access_cucina = bool(request.form.get("access_cucina"))
            if not areas_selected:
                if access_sala:
                    areas_selected.append("sala")
                if access_cucina:
                    areas_selected.append("cucina")

            if not (username and password):
                flash("Username e password sono obbligatori.", "danger")
                return redirect(url_for("admin_departments"))

            if User.query.filter_by(username=username).first():
                flash("Username già esistente.", "danger")
                return redirect(url_for("admin_departments"))

            if role == "admin":
                u = User(
                    username=username,
                    display_name=username,   # usa username come display
                    role="admin",
                    admin_level=("partial" if admin_level == "partial" else "full"),
                    works_for_client_id=None,
                    access_sala=True,
                    access_cucina=True,
                )
                u.set_password(password)
                db.session.add(u)
                db.session.commit()
                flash("Utente admin creato.", "success")
                return redirect(url_for("admin_departments"))

            # default: employee (utente negozio)
            if not shop_id:
                flash("Seleziona un negozio per l'utente di tipo 'Utente negozio'.", "danger")
                return redirect(url_for("admin_departments"))

            store = User.query.filter_by(id=shop_id, role="client").first()
            if not store:
                flash("Negozio non valido.", "danger")
                return redirect(url_for("admin_departments"))

            emp = User(
                username=username,
                display_name=username,
                role="employee",
                works_for_client_id=shop_id,
                # legacy flags: manteniamo compatibilità per Sala/Cucina
                access_sala=("sala" in areas_selected),
                access_cucina=("cucina" in areas_selected),
            )
            emp.set_password(password)
            db.session.add(emp)
            db.session.commit()

            # Salva accessi macro-area (tabella pivot)
            # Normalizza + valida
            valid_areas = set(list_macro_areas())
            to_link = sorted({a.strip().lower() for a in areas_selected if a and a.strip().lower() in valid_areas})

            for area in to_link:
                # evita duplicati se esiste unique(user_id, area)
                if not UserMacroAccess.query.filter_by(user_id=emp.id, area=area).first():
                    db.session.add(UserMacroAccess(user_id=emp.id, area=area))
            db.session.commit()

            flash("Utente negozio creato.", "success")
            return redirect(url_for("admin_departments"))

        # tipo form inatteso → fallback
        return redirect(url_for("admin_departments"))

    # --- GET ---
    departments = Department.query.order_by(Department.name).all()
    current_sala_code   = get_macro_code("sala")
    current_cucina_code = get_macro_code("cucina")
    shops = User.query.filter_by(role="client").order_by(User.display_name).all()

    admins    = User.query.filter_by(role="admin").order_by(User.display_name).all()
    employees = User.query.filter_by(role="employee").order_by(User.display_name).all()
    all_users = admins + employees
    all_users.sort(key=lambda u: (0 if u.role=='admin' else 1, (u.display_name or u.username or "").lower()))

    users = (
        User.query
        .filter(User.role.in_(["admin", "employee"]))
        .order_by(User.role.desc(), User.display_name.asc())
        .all()
    )

    # <<< QUI LA FIX: nessun MacroArea model >>>
    from types import SimpleNamespace
    # prendi le macroaree come lista di stringhe dall'helper esistente
    areas_list = sorted(set(list_macro_areas() or []))
    # trasformale in oggetti con attributo .area così il template funziona (m.area)
    macro_areas = [SimpleNamespace(area=a) for a in areas_list]

    # Mappa user -> [aree] per le badge nel template
    rows = db.session.query(UserMacroAccess.user_id, UserMacroAccess.area).all()
    user_area_map = {}
    for uid, area in rows:
        user_area_map.setdefault(uid, []).append(area)

    return render_template(
        "admin/departments.html",
        departments=departments,
        shops=shops,
        all_users=all_users,
        current_sala_code=current_sala_code,
        current_cucina_code=current_cucina_code,
        users=users,
        macro_areas=macro_areas,     # ora è una lista di SimpleNamespace(area=...)
        user_area_map=user_area_map, # usato dal template per le badge
    )



@app.route("/admin/prodotti", methods=["GET", "POST"])
@login_required
@require_role("admin")
def admin_products():
    departments = Department.query.order_by(Department.name).all()
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        dep_id = request.form.get("department_id")
        if not (name and dep_id):
            flash("Nome e reparto sono obbligatori.", "danger")
        else:
            p = Product(name=name, department_id=int(dep_id), active=True)  # niente SKU
            db.session.add(p)
            db.session.commit()
            flash("Prodotto creato.", "success")
        return redirect(url_for("admin_products"))
    products = Product.query.order_by(Product.name).all()
    return render_template("admin/products.html", products=products, departments=departments)

# --- UPDATE endpoints: client, reparti, prodotti -----------------------------

@app.post("/admin/clients/<int:client_id>/update")
@login_required
@require_role("admin")
def admin_update_client(client_id):
    c = User.query.get_or_404(client_id)
    if c.role != "client":
        flash("Puoi modificare solo utenti di tipo 'client'.", "danger")
        return redirect(url_for("admin_clients"))

    new_display  = (request.form.get("display_name") or "").strip()
    new_username = (request.form.get("username") or "").strip()  # opzionale, il form tuo non lo invia

    if not new_display:
        flash("Nome punto vendita richiesto.", "danger")
        return redirect(url_for("admin_clients"))

    # Se qualcuno fornisse anche uno username diverso, verifica unicità e applicalo
    if new_username and new_username != c.username:
        exists = User.query.filter(User.username == new_username, User.id != c.id).first()
        if exists:
            flash("Username già esistente.", "danger")
            return redirect(url_for("admin_clients"))
        c.username = new_username  # altrimenti lascia quello attuale

    c.display_name = new_display
    db.session.commit()
    flash("Client aggiornato.", "success")
    return redirect(url_for("admin_clients"))

@app.post("/admin/reparti/<int:dep_id>/update")
@login_required
@require_role("admin")
def admin_update_department(dep_id):
    d = Department.query.get_or_404(dep_id)
    new_name = (request.form.get("name") or "").strip()
    new_area = (request.form.get("macro_area") or "").strip().lower()
    if new_area not in set(list_macro_areas()):
        flash("Macro-area non valida.", "danger")
        return redirect(url_for("admin_departments"))
    d.macro_area = new_area

    if not new_name:
        flash("Nome reparto richiesto.", "danger")
        return redirect(url_for("admin_departments"))

    exists = Department.query.filter(Department.name == new_name, Department.id != d.id).first()
    if exists:
        flash("Esiste già un reparto con questo nome.", "danger")
        return redirect(url_for("admin_departments"))

    d.name = new_name
    d.macro_area = new_area
    db.session.commit()
    flash("Reparto aggiornato.", "success")
    return redirect(url_for("admin_departments"))


@app.post("/admin/prodotti/<int:prod_id>/update")
@login_required
@require_role("admin")
def admin_update_product(prod_id):
    p = Product.query.get_or_404(prod_id)
    new_name = (request.form.get("name") or "").strip()
    dep_id   = request.form.get("department_id", type=int)

    if not new_name or not dep_id:
        flash("Nome e reparto sono obbligatori.", "danger")
        return redirect(url_for("admin_products"))

    dep = Department.query.get(dep_id)
    if not dep:
        flash("Reparto non valido.", "danger")
        return redirect(url_for("admin_products"))

    p.name = new_name
    p.department_id = dep_id
    db.session.commit()
    flash("Prodotto aggiornato.", "success")
    return redirect(url_for("admin_products"))

# --- DELETE endpoints: reparti e prodotti ---
@app.post("/admin/reparti/<int:dep_id>/delete")
@login_required
@require_role("admin")
def admin_delete_department(dep_id):
    if is_partial_admin():
        flash("Non autorizzato: questo account admin non può eliminare reparti.", "danger")
        return redirect(url_for("admin_departments"))
    d = Department.query.get_or_404(dep_id)
    if len(d.products) > 0:
        flash("Impossibile eliminare: il reparto contiene ancora prodotti.", "danger")
        return redirect(url_for("admin_departments"))
    db.session.delete(d)
    db.session.commit()
    flash("Reparto eliminato.", "success")
    return redirect(url_for("admin_departments"))

@app.post("/admin/prodotti/<int:prod_id>/delete")
@login_required
@require_role("admin")
def admin_delete_product(prod_id):
    if is_partial_admin():
        flash("Non autorizzato: questo account admin non può eliminare prodotti.", "danger")
        return redirect(url_for("admin_products"))
    p = Product.query.get_or_404(prod_id)
    # rimuove eventuali righe richiesta collegate a questo prodotto
    RequestItem.query.filter_by(product_id=p.id).delete(synchronize_session=False)
    # ✱✱ NUOVO: rimuovi anche le righe dei piani di distribuzione collegate ✱✱
    DistributionLine.query.filter_by(product_id=p.id).delete(synchronize_session=False)
    db.session.delete(p)
    db.session.commit()
    flash("Prodotto eliminato.", "success")
    return redirect(url_for("admin_products"))


@app.post("/admin/clients/<int:client_id>/delete")
@login_required
@require_role("admin")
def admin_delete_client(client_id):
    if is_partial_admin():
        flash("Non autorizzato: questo account admin non può eliminare negozi.", "danger")
        return redirect(url_for("admin_clients"))
    # Non permettere di eliminare se stessi o utenti non-client
    if current_user.id == client_id:
        flash("Non puoi eliminare te stesso.", "danger")
        return redirect(url_for("admin_clients"))

    client = User.query.get_or_404(client_id)
    if client.role != "client":
        flash("Puoi eliminare solo utenti di tipo 'client'.", "danger")
        return redirect(url_for("admin_clients"))


    req_ids = [rid for (rid,) in db.session.query(RequestHeader.id).filter_by(client_id=client.id).all()]
    if req_ids:
        RequestItem.query.filter(RequestItem.request_id.in_(req_ids)).delete(synchronize_session=False)
        RequestHeader.query.filter(RequestHeader.id.in_(req_ids)).delete(synchronize_session=False)

    DistributionLine.query.filter_by(client_id=client.id).delete(synchronize_session=False)

    db.session.delete(client)
    db.session.commit()

    flash("Client eliminato.", "success")
    return redirect(url_for("admin_clients"))


@app.route("/admin/giacenze")
@login_required
@require_role("admin")
def admin_stock_totals():
    # filtro testo facoltativo
    q = (request.args.get("q") or "").strip().lower()

    shops = User.query.filter_by(role="client").order_by(User.display_name).all()
    areas = list_macro_areas() or ["sala", "cucina"]  # dinamico + fallback

    # product_id -> {"product": Product, "total": Decimal, "breakdown": {client_name: Decimal}}
    totals = {}

    def _accumulate_from(rh: RequestHeader, area_norm: str, shop_name: str):
        if not rh:
            return
        for it in rh.items:
            qty = it.qty_requested or 0
            if qty <= 0:
                continue
            if not it.product or not it.product.department:
                continue
            if (it.product.department.macro_area or "sala").lower() != area_norm:
                continue
            pid = it.product_id
            rec = totals.setdefault(pid, {"product": it.product, "total": 0, "breakdown": {}})
            rec["total"] += qty
            rec["breakdown"][shop_name] = rec["breakdown"].get(shop_name, 0) + qty

    # per ogni negozio prendo l'ULTIMA richiesta con righe >0 per ciascuna macro-area
    for shop in shops:
        shop_name = shop.display_name
        for area in areas:
            area_norm = (area or "sala").lower()
            last_in_area = (
                RequestHeader.query
                .filter_by(client_id=shop.id, status="inviata")
                .join(RequestItem, RequestItem.request_id == RequestHeader.id)
                .join(Product, Product.id == RequestItem.product_id)
                .join(Department, Department.id == Product.department_id)
                .filter(func.lower(Department.macro_area) == area_norm,
                        RequestItem.qty_requested > 0)
                .order_by(RequestHeader.created_at.desc())
                .first()
            )
            _accumulate_from(last_in_area, area_norm, shop_name)

    # ordina per reparto, poi prodotto
    rows = sorted(
        totals.values(),
        key=lambda r: (
            (r["product"].department.name.lower() if r["product"] and r["product"].department else ""),
            (r["product"].name.lower() if r["product"] and r["product"].name else "")
        ),
    )

    # filtro testo su nome prodotto
    if q:
        rows = [r for r in rows if q in (r["product"].name or "").lower()]

    return render_template("admin/stock_totals.html", rows=rows, q=(request.args.get("q") or ""))


@app.route("/admin/macro-codici", methods=["GET", "POST"])
@login_required
@require_role("admin")
def admin_macro_codes():
    sala = MacroAreaAccess.query.filter_by(area="sala").first()
    cucina = MacroAreaAccess.query.filter_by(area="cucina").first()
    if request.method == "POST":
        if is_partial_admin():
            flash("Non autorizzato: questo account admin non può creare prodotti.", "danger")
            return redirect(url_for("admin_products"))
        sala_code = (request.form.get("sala_code") or "").strip()
        cucina_code = (request.form.get("cucina_code") or "").strip()
        if sala is None:
            sala = MacroAreaAccess(area="sala", code=sala_code)
            db.session.add(sala)
        else:
            sala.code = sala_code
        if cucina is None:
            cucina = MacroAreaAccess(area="cucina", code=cucina_code)
            db.session.add(cucina)
        else:
            cucina.code = cucina_code
        db.session.commit()
        flash("Codici aggiornati.", "success")
        return redirect(url_for("admin_macro_codes"))
    return render_template("admin/macro_codes.html", sala=sala, cucina=cucina)

@app.post("/client/macro/unlock")
@login_required
@require_role("employee")
def client_macro_unlock():
    area = (request.form.get("area") or "").strip().lower()
    code = (request.form.get("code") or "").strip()

    # Validazione dinamica
    known = list_macro_areas()
    if area not in known:
        flash("Area non valida.", "danger")
        return redirect(url_for("client_departments"))

    expected = get_macro_code(area)  # "" = nessun codice richiesto
    if expected and code != expected:
        flash("Codice errato.", "danger")
        return redirect(url_for("client_departments"))

    unlocked = session.get("macro_unlock", {})
    unlocked[area] = True
    session["macro_unlock"] = unlocked
    flash(f"Area '{area}' sbloccata!", "success")
    return redirect(url_for("client_departments"))

@app.route("/client/ricerca", methods=["GET", "POST"])
@login_required
@require_roles("employee")
def client_search():
    q = request.args.get("q", "").strip()
    client_id_for_work = current_user.works_for_client_id
    draft = get_or_create_draft_request(client_id_for_work)

    allowed = allowed_macro_areas_for(current_user)
    if not allowed:
        flash("Nessuna macro-area abilitata per il tuo account.", "warning")
        return redirect(url_for("client_departments"))

    products = []
    min_len = 2
    if q and len(q) >= min_len:
        products = (
            Product.query
            .join(Department, Product.department_id == Department.id)
            .filter(Product.active == True, Department.active == True)
            .filter(func.lower(Product.name).like(f"%{q.lower()}%"))
            .order_by(Department.name, Product.name)
            .all()
        )

    if request.method == "POST":
        q_post = request.form.get("q", "").strip()
        post_products = []
        if q_post and len(q_post) >= min_len:
            post_products = (
                Product.query
                .join(Department, Product.department_id == Department.id)
                .filter(Product.active == True, Department.active == True)
                .filter(func.lower(Product.name).like(f"%{q_post.lower()}%"))
                .order_by(Department.name, Product.name)
                .all()
            )
        for p in post_products:
            from decimal import Decimal
            qty_str = (request.form.get(f"qty_{p.id}", "0") or "").strip()
            qty = parse_decimal(qty_str).quantize(Decimal("0.001"))
            if qty < 0:
                qty = Decimal("0")
            note = request.form.get(f"note_{p.id}", "").strip() or None
            item = next((i for i in draft.items if i.product_id == p.id), None)
            if item is None:
                item = RequestItem(request_id=draft.id, product_id=p.id, qty_requested=qty, note=note)
                db.session.add(item)
            else:
                item.qty_requested = qty
                item.note = note
        db.session.commit()
        flash("Quantità salvate dai risultati di ricerca.", "success")
        return redirect(url_for("client_search", q=q_post) if q_post else url_for("client_search"))

    qty_map = {i.product_id: i for i in draft.items}
    return render_template("client/search.html", q=q, products=products, qty_map=qty_map, min_len=min_len)

@app.route("/admin/griglia", methods=["GET", "POST"])
@login_required
@require_role("admin")
def admin_grid():
    # --- anagrafiche base ---
    clients = (User.query
               .filter_by(role="client")
               .order_by(User.display_name)
               .all())

    # Reparti + selezione reparto corrente (default: primo reparto attivo per nome)
    departments = (Department.query
                   .filter_by(active=True)
                   .order_by(Department.name)
                   .all())
    if not departments:
        selected_dep_id = None
        products = []
    else:
        selected_dep_id = request.args.get("dep_id", type=int) or departments[0].id
        products = (Product.query
                    .filter_by(active=True, department_id=selected_dep_id)
                    .order_by(Product.name)
                    .all())

    # Flag per mostrare il recap SOLO quando richiesto esplicitamente
    show_recap = request.args.get("show_recap") == "1"

    # Ultima giacenza inviata per ogni client NELL'AREA CORRENTE (Ricevuta)
    selected_area = None
    if selected_dep_id:
        dep_sel = next((d for d in departments if d.id == selected_dep_id), None)
        selected_area = (dep_sel.macro_area if dep_sel else "sala")
    else:
        selected_area = "sala"

    received = {c.id: {} for c in clients}

    for shop in clients:
        last_req_in_area = (
            RequestHeader.query
            .filter_by(client_id=shop.id, status="inviata")
            .join(RequestItem, RequestItem.request_id == RequestHeader.id)
            .join(Product, Product.id == RequestItem.product_id)
            .join(Department, Department.id == Product.department_id)
            .filter(Department.macro_area == selected_area, RequestItem.qty_requested > 0)
            .order_by(RequestHeader.created_at.desc())
            .first()
        )
        if not last_req_in_area:
            continue

        for it in last_req_in_area.items:
            if (it.qty_requested or 0) <= 0:
                continue
            if not it.product or not it.product.department:
                continue
            if (it.product.department.macro_area or "sala") != selected_area:
                continue
            # se qty_requested è Decimal, va benissimo così; se vuoi forzare float: float(it.qty_requested)
            received[shop.id][it.product_id] = it.qty_requested

    # --- quantità consigliate per (client, product) ---
    _recs = ClientProductRecommendation.query.all()
    recommended = {(r.client_id, r.product_id): r.recommended_qty for r in _recs}

    # --- piano attivo / bozza / storico ---
    view_plan_id = request.args.get("plan_id", type=int)
    draft = get_or_create_draft_plan(current_user.id)
    active_plan = draft
    read_only = False
    viewing_message = None

    if view_plan_id and view_plan_id != draft.id:
        plan_to_view = DistributionPlan.query.get_or_404(view_plan_id)
        active_plan = plan_to_view
        read_only = True
        viewing_message = (
        f"Stai visualizzando il piano #{plan_to_view.id} ({plan_to_view.status}) "
        f"creato il {to_rome(plan_to_view.created_at).strftime('%Y-%m-%d %H:%M')}."
    )

    # --- POST: salva bozza / recap / conferma ---
    if request.method == "POST":
        action = request.form.get("action", "save")
        target = draft

        # capisco se questo POST contiene gli input quantità
        has_qty_fields = any(k.startswith("in_") or k.startswith("out_") for k in request.form.keys())

        # Ricostruisci/aggiorna righe SOLO per il reparto selezionato
        if has_qty_fields:
            for c in clients:
                for p in products:  # SOLO prodotti del reparto selezionato
                    q_in_str  = request.form.get(f"in_{c.id}_{p.id}", "0")
                    q_out_str = request.form.get(f"out_{c.id}_{p.id}", "0")

                    q_in  = parse_decimal(q_in_str)
                    q_out = parse_decimal(q_out_str)
                    # mutua esclusione
                    if q_in > 0:
                        q_out = Decimal("0")
                    elif q_out > 0:
                        q_in = Decimal("0")

                    ln = (DistributionLine.query
                          .filter_by(plan_id=target.id, client_id=c.id, product_id=p.id)
                          .first())

                    if (q_in > 0) or (q_out > 0):
                        if ln is None:
                            ln = DistributionLine(
                                plan_id=target.id, client_id=c.id, product_id=p.id,
                                qty_in=q_in, qty_out=q_out
                            )
                            db.session.add(ln)
                        else:
                            ln.qty_in = q_in
                            ln.qty_out = q_out
                    else:
                        if ln is not None:
                            db.session.delete(ln)

            db.session.commit()

        # Per i redirect preservo sempre il reparto corrente
        dep_for_redirect = (request.form.get("dep_id")
                            or request.args.get("dep_id")
                            or str(selected_dep_id) if selected_dep_id else None)

        # Gestione azione
        if action == "confirm":
            target.status = "confermato"
            db.session.commit()
            flash("Piano confermato.", "success")
            # Torna alla griglia sullo stesso reparto (nessun recap automatico)
            return redirect(url_for("admin_grid", dep_id=dep_for_redirect))

        elif action == "recap":
            flash("Bozza salvata. Recap in basso.", "info")
            # Mostra esplicitamente il recap (show_recap=1) e resta sul reparto selezionato
            return redirect(url_for("admin_grid", dep_id=dep_for_redirect, show_recap=1))

        else:  # "save"
            flash("Bozza salvata.", "success")
            # Nessun recap automatico dopo il salvataggio
            return redirect(url_for("admin_grid", dep_id=dep_for_redirect))

    # mappa valori in/out per pre-popolare gli input
    plan_map = {}
    if active_plan:
        for ln in active_plan.lines:
            plan_map[(ln.client_id, ln.product_id)] = {"in": float(ln.qty_in), "out": float(ln.qty_out)}

    # --- Evidenzia negozi "verdi" finché non arriva una nuova giacenza ---
    # Ultimo PIANO confermato che coinvolge il client (almeno una riga >0)
    last_conf_rows = (
        db.session.query(
            DistributionLine.client_id,
            func.max(DistributionPlan.created_at).label("ts")
        )
        .join(DistributionPlan, DistributionPlan.id == DistributionLine.plan_id)
        .join(Product, Product.id == DistributionLine.product_id)
        .filter(DistributionPlan.status == "confermato")
        .filter(Product.department_id == selected_dep_id)
        .filter(DistributionLine.qty_in > 0)  # conteggia anche auto-fill (qty_in salvato)
        .group_by(DistributionLine.client_id)
        .all()
    )
    last_conf_by_client = {cid: ts for cid, ts in last_conf_rows}

    # Ultima GIACENZA inviata dal client nella macro-area del reparto selezionato
    last_req_rows = (
        db.session.query(
            RequestHeader.client_id,
            func.max(RequestHeader.created_at).label("ts")
        )
        .join(RequestItem, RequestItem.request_id == RequestHeader.id)
        .join(Product, Product.id == RequestItem.product_id)
        .join(Department, Department.id == Product.department_id)
        .filter(RequestHeader.status == "inviata")
        .filter(RequestItem.qty_requested > 0)
        .filter(Department.macro_area == selected_area)
        .group_by(RequestHeader.client_id)
        .all()
    )
    last_req_by_client = {cid: ts for cid, ts in last_req_rows}

    # Verde se: esiste un piano confermato per questo reparto più recente dell’ultima giacenza di quell’area
    highlight_clients = {
        c.id for c in clients
        if (last_conf_by_client.get(c.id)
            and (not last_req_by_client.get(c.id)
                or last_conf_by_client[c.id] > last_req_by_client[c.id]))
    }

        
    # --- Recap robusto (per piano specifico) ---
    def build_recap(plan_id: int):
        from collections import defaultdict
        from decimal import Decimal

        zero = Decimal("0")
        lines = (
            DistributionLine.query
            .filter(DistributionLine.plan_id == plan_id)
            .filter((DistributionLine.qty_in > zero) | (DistributionLine.qty_out > zero))
            .options(
                joinedload(DistributionLine.product),
                joinedload(DistributionLine.client),
            )
            .all()
        )

        by_prod = defaultdict(list)
        for ln in lines:
            by_prod[ln.product_id].append(ln)

        rows = []
        for pid, lns in by_prod.items():
            prod = lns[0].product if lns else Product.query.get(pid)
            if not prod:
                continue
            tot_in = sum(x.qty_in for x in lns)
            tot_out = sum(x.qty_out for x in lns)
            breakdown = [{"client": x.client.display_name, "in": x.qty_in, "out": x.qty_out} for x in lns]
            rows.append({"product": prod, "tot_in": tot_in, "tot_out": tot_out, "breakdown": breakdown})

        rows.sort(key=lambda r: r["product"].name.lower())
        return rows

    recap_rows = build_recap(active_plan.id) if active_plan else []


    # storico ultimi piani
    history = (
        DistributionPlan.query
        .options(
            joinedload(DistributionPlan.lines).joinedload(DistributionLine.product).joinedload(Product.department),
            joinedload(DistributionPlan.lines).joinedload(DistributionLine.client),
        )
        .order_by(DistributionPlan.created_at.desc())
        .limit(12)
        .all()
    )
    history_recaps = {pl.id: build_recap(pl.id) for pl in history}

    # Ultimo recap confermato
    latest_confirmed = (
        DistributionPlan.query
        .filter(DistributionPlan.status == "confermato")
        .order_by(DistributionPlan.created_at.desc())
        .first()
    )
    latest_confirmed_recap = build_recap(latest_confirmed.id) if latest_confirmed else []

    # Filtri data (per recap multipli sotto)
    from_str = request.args.get("from", "").strip()
    to_str   = request.args.get("to", "").strip()

    date_from = None
    date_to   = None
    try:
        if from_str:
            date_from = datetime.strptime(from_str, "%Y-%m-%d").date()
    except ValueError:
        date_from = None
    try:
        if to_str:
            date_to = datetime.strptime(to_str, "%Y-%m-%d").date()
    except ValueError:
        date_to = None

    filtered_plans = []
    filtered_recaps = {}
    if date_from or date_to:
        q = DistributionPlan.query.filter(DistributionPlan.status == "confermato")
        if date_from:
            q = q.filter(func.date(DistributionPlan.created_at) >= date_from.strftime("%Y-%m-%d"))
        if date_to:
            q = q.filter(func.date(DistributionPlan.created_at) <= date_to.strftime("%Y-%m-%d"))
        filtered_plans = q.order_by(DistributionPlan.created_at.desc()).limit(200).all()
        filtered_recaps = {pl.id: build_recap(pl.id) for pl in filtered_plans}

    return render_template(
        "admin/grid.html",
        # base
        clients=clients,
        products=products,
        received=received,
        plan_map=plan_map,
        active_plan=active_plan,
        read_only=read_only,
        viewing_message=viewing_message,
        # recap
        recap_rows=recap_rows,
        history=history,
        history_recaps=history_recaps,
        latest_confirmed=latest_confirmed,
        latest_confirmed_recap=latest_confirmed_recap,
        # filtri recap by date
        date_from=from_str,
        date_to=to_str,
        filtered_plans=filtered_plans,
        filtered_recaps=filtered_recaps,
        # reparti
        departments=departments,
        selected_dep_id=selected_dep_id,
        # flag recap esplicito
        show_recap=show_recap,
        recommended=recommended,
        highlight_clients=highlight_clients,
    )

@app.post("/admin/griglia/set_recommended")
@login_required
@require_role("admin")
def admin_grid_set_recommended():
    data = request.get_json(force=True, silent=True) or {}
    try:
        client_id = int(data.get("client_id", 0))
        product_id = int(data.get("product_id", 0))
        value = int(data.get("value", 0))
        value = max(0, value)
    except Exception:
        return jsonify(ok=False, error="Parametri non validi"), 400

    # trova o crea
    rec = ClientProductRecommendation.query.filter_by(
        client_id=client_id, product_id=product_id
    ).first()

    old = rec.recommended_qty if rec else 0

    if rec is None:
        rec = ClientProductRecommendation(
            client_id=client_id, product_id=product_id, recommended_qty=value
        )
        db.session.add(rec)
    else:
        rec.recommended_qty = value

    db.session.commit()
    return jsonify(ok=True, old=old, value=value)


from decimal import Decimal, InvalidOperation

@app.post("/admin/griglia/save-ajax")
@login_required
@require_role("admin")
def admin_grid_save_ajax():
    """
    Salva una singola cella (IN) del piano bozza dell'admin corrente.
    Body JSON: { client_id, product_id, kind: "in"|"out", value: number }
    """
    data = request.get_json(silent=True) or {}
    try:
        client_id  = int(data.get("client_id"))
        product_id = int(data.get("product_id"))
        kind       = str(data.get("kind") or "in")
        value      = parse_decimal(data.get("value"))
    except (ValueError, TypeError):
        return jsonify({"ok": False, "error": "Bad payload"}), 400

    plan = get_or_create_draft_plan(current_user.id)

    ln = DistributionLine.query.filter_by(
        plan_id=plan.id, client_id=client_id, product_id=product_id
    ).first()

    if ln is None:
        ln = DistributionLine(
            plan_id=plan.id, client_id=client_id, product_id=product_id,
            qty_in=Decimal("0"), qty_out=Decimal("0")
        )
        db.session.add(ln)

    zero = Decimal("0")

    if kind == "in":
        ln.qty_in = value
        if value > zero:
            ln.qty_out = zero
    else:
        ln.qty_out = value
        if value > zero:
            ln.qty_in = zero

    # se entrambe zero, puliamo la riga
    if (ln.qty_in or zero) == zero and (ln.qty_out or zero) == zero:
        db.session.delete(ln)
        db.session.commit()
        return jsonify({"ok": True, "qty_in": "0", "qty_out": "0"})

    db.session.commit()
    return jsonify({
        "ok": True,
        "qty_in": str(ln.qty_in),
        "qty_out": str(ln.qty_out)
    })

# ---------- CLI ----------
@app.cli.command("init-db")
def init_db_command():
    """Inizializza il DB e crea dati demo."""
    db.create_all()
    ensure_seed_data()
    print("DB inizializzato con dati demo. Utenti: admin/admin, client1/client1, client2/client2")

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        ensure_schema_upgrade()
        ensure_seed_data()
    port = int(os.environ.get("PORT", "5500"))
    app.run(debug=True, host="0.0.0.0", port=port)

@app.cli.command("reset-db")
def reset_db_command():
    """Drop & recreate tables, then seed demo data."""
    db.drop_all()
    db.create_all()
    ensure_seed_data()
    print("DB resettato e ripopolato.")

    
@app.context_processor
def inject_pending_draft_info():
    is_emp = current_user.is_authenticated and getattr(current_user, "role", None) == "employee"
    count = _draft_filled_rows_count_for_current_employee() if is_emp else 0

    is_admin = current_user.is_authenticated and getattr(current_user, "role", None) == "admin"
    level = (getattr(current_user, "admin_level", "full") or "full") if is_admin else None

    return {
        "is_employee": is_emp,
        "has_pending_draft_qty": is_emp and count > 0,
        "pending_draft_qty_count": count,
        # --- NUOVI FLAG ---
        "is_admin_full":    bool(is_admin and level == "full"),
        "is_admin_partial": bool(is_admin and level == "partial"),
    }