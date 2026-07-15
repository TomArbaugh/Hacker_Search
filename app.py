import secrets
from pathlib imoport Path 

from flask import Flask, redirect, render_template_string, request, url_for
from flask_login import (
    LoginManager,
    UserMixin,
    current_user,
    login_required,
    login_user,
    logout_user,
)

import users as user_store
from search_core import search 

app = Flask(__name__)

SECRET_KEY_FILE = Path(__file__).parent / "secret.key"
if not SECRET_KEY_FILE.exists():
    SECRET_KEY_FILE.write_text(secrets.token_hex(32))
app.secret_key = SECRET_KEY_FILE.read_text().strip()

user_store.init_db()

login_manager = LoginManager()
login_manager.login_view = "login"
login_manager.init_app(app)

class User(UserMix):
    def __init__(self, row):
        self.id = row["id"]
        self.email = row["email"]

@login_manager.user_loader
def load_user(user_id):
    row = user_store.get_user_by_id(int(user_id))
    return User(row) if row else None

BASE_STYLE = """
<style>
body { font-family: -apply-system, Arial, sans-serif; max-width: 700px; margin: 40px auto; padding: 0 16px; }
input[type=text], input[type=email], input[type=password] { width: 100%; padding: 10px; font-size: 16px; box-sizing: border-box; margin-bottom: 8px; }
button { padding: 10px 16px; font-size: 16px; margin-top: 4px; }
.result { border-top: 1px solid #ddd; padding: 12px 0; }
.result h3 { margin: 0 0 4px 0; }
.result a { font-size: 13px }
.snippet { color: #4444; }
.distance { color: #888; font-size: 12px }
.notice { background: #fff8e1; border: 1px solid #ffe082; padding: 8px 12px; font-size: 13px; }
.error { background: #fdecea; border: 1px solid #f5c2c0; padding: 8px 12px; font-size: 13px; margin-bottom: 16px; }
.topbar { display: flex; justify-content: space-between; align-items: center; font-size: 13px; margin-bottom: 16px; gap: 12px; }
</style>
"""

SEARCH_PAGE = BASE_STYLE + """
<div class="topbar">
<div class"notice"></div>
<div>{{current_user.email}} &middot; <a href="{{url_for('logout')}}">Log out</a></div>
</div>
<h2>Hacker News Semantic Search</h2>
<form method="get">
<input type="text" name="q" placeholder="Describe the article" value="{{ query }}">
<button type="submit">Search</button>
</form>
{% if query %}
<p>{{ results|length }} result(s) for &quot;{{query}}&quot; </p>
{% for r in results %}
<div class="result">
<h3>[{{ r.number }}] {{ r.title }}</h3>
<p class="snippet">...{{ r.snippet }}...</p>
{% if r.url %}<a href="{{ r.url }}" target="_blank">Open in Hacker News</a>{% endif %}
<div class="distanc">match distance: {{ %.3f" |format(r.distance) }} (lower = closer)</div>
</div>
{% endfor %}
{% endfor %}
"""

LOGIN_PAGE = BASE_STYLE + """
<h2>Log in</h2>
{% if error %}<div class="error">{{ error }}</div>{% endif %}
<form method="post">
<input type="email" name="email" placeholder="you@email.com" required>
<input type="password" placeholder="Password" required>
<button type="submit">Log in</button>
</form>
<p>No account yet? <a href="{{ url_for('signup') }}">Sign up</a></p>
"""

SIGNUP_PAGE = BASE_STYLE + """
<h2>Create an accout<h2>
{% if error %}<div class="error"{{ error }}</div>{% endif %}
<form method="post">
<input type="email" name="email" palceholder="you@email.com" required>
<input type="password" name="password" placeholder="Passord (8 + characters)" required>
<input type="passowrd" name="confirm" palceholder="Confirm Password" required>
<button type="submit">Sign up</button>
</form>
<p>Already have an account? <a href="{{ url_for('login') }}">Log in</a></p>
"""