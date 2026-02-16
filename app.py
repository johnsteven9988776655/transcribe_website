from flask import Flask, request, render_template, send_file, jsonify, redirect, url_for, flash, session
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Length, EqualTo
from werkzeug.security import generate_password_hash, check_password_hash
import whisper
from pydub import AudioSegment
import os
import math
import uuid
import tempfile
from werkzeug.utils import secure_filename
import threading
import time
from datetime import datetime
from functools import wraps
import platform
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-this-in-production'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///transcriptions.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'transcriptions'

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Store transcription status (in-memory for simplicity)
transcription_status = {}

# Set FFmpeg paths
ffmpeg_path = r"C:\ffmpeg-8.0-essentials_build\bin\ffmpeg.exe"
ffprobe_path = r"C:\ffmpeg-8.0-essentials_build\bin\ffprobe.exe"
AudioSegment.converter = ffmpeg_path
AudioSegment.ffmpeg = ffmpeg_path
AudioSegment.ffprobe = ffprobe_path
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg-8.0-essentials_build\bin"

# Local Whisper models directory
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    transcriptions = db.relationship('Transcription', backref='user', lazy=True, cascade='all, delete-orphan')
    is_admin = db.Column(db.Boolean, default=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Transcription(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    job_id = db.Column(db.String(36), unique=True, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    model_size = db.Column(db.String(20), nullable=False)
    status = db.Column(db.String(20), default='processing')
    duration = db.Column(db.String(50))
    output_file = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime)
    file_size = db.Column(db.Integer)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Forms
class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=4, max=20)])
    email = StringField('Email', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    password2 = PasswordField('Confirm Password', 
                             validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Register')

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Sign In')

class ResetPasswordForm(FlaskForm):
    password = PasswordField('New Password', validators=[DataRequired(), Length(min=6)])
    password2 = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Reset Password')

def allowed_file(filename):
    """Check if file extension is allowed"""
    ALLOWED_EXTENSIONS = {'mp3', 'wav', 'mp4', 'avi', 'm4a', 'flac', 'ogg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Add a watchdog timeout (in minutes) for long-running jobs
MAX_TRANSCRIBE_MINUTES = 10

def transcribe_audio_file(file_path, job_id, model_size="base", chunk_minutes=8):
    """Background function to transcribe audio file with improved preprocessing"""
    try:
        # Use application context for all database operations
        with app.app_context():
            # Update database status
            transcription = Transcription.query.filter_by(job_id=job_id).first()
            if transcription:
                transcription.status = 'processing'
                db.session.commit()

        transcription_status[job_id]['status'] = 'processing'
        transcription_status[job_id]['progress'] = 'Loading Whisper model...'
        
        try:
            # Load normally (uses network on first run, then local cache)
            model = whisper.load_model(model_size)
        except Exception as e:
            # Surface clearer error and stop early
            transcription_status[job_id]['status'] = 'error'
            transcription_status[job_id]['progress'] = f'Error loading model: {e}. Ensure internet access for first-time download.'
            with app.app_context():
                t = Transcription.query.filter_by(job_id=job_id).first()
                if t:
                    t.status = 'error'
                    db.session.commit()
            return
        
        transcription_status[job_id]['progress'] = 'Converting audio format...'
        
        # Convert to WAV for processing
        audio = AudioSegment.from_file(file_path)
        # Audio preprocessing: mono, 16kHz, normalized
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(16000)
        audio = audio.normalize()
        temp_wav = os.path.join(tempfile.gettempdir(), f"temp_{job_id}.wav")
        audio.export(temp_wav, format="wav")
        
        # Get audio duration
        duration_seconds = len(audio) / 1000.0
        duration_minutes = duration_seconds / 60.0
        
        transcription_status[job_id]['duration'] = f"{duration_minutes:.1f} minutes"
        
        # Update database with duration
        with app.app_context():
            transcription = Transcription.query.filter_by(job_id=job_id).first()
            if transcription:
                transcription.duration = f"{duration_minutes:.1f} minutes"
                db.session.commit()
        
        # Determine if we need chunking
        chunk_seconds = chunk_minutes * 60
        
        # Watchdog start time
        start_ts = time.time()

        def transcribe_path(path):
            # Force English decoding for stronger phonetic transcription
            return model.transcribe(
                path,
                task='transcribe',
                language='en',
                temperature=0.0,
            )

        if duration_seconds <= chunk_seconds:
            # Process entire file
            transcription_status[job_id]['progress'] = 'Transcribing audio...'
            result = transcribe_path(temp_wav)
            transcribed_text = result.get("text", "")
        else:
            # Process in chunks
            num_chunks = math.ceil(duration_seconds / chunk_seconds)
            transcription_status[job_id]['total_chunks'] = num_chunks
            transcribed_chunks = []
            
            for i in range(num_chunks):
                # Watchdog: abort if taking too long
                if (time.time() - start_ts) > (MAX_TRANSCRIBE_MINUTES * 60):
                    raise Exception('Transcription timed out')
                start_time = i * chunk_seconds * 1000
                end_time = min((i + 1) * chunk_seconds * 1000, len(audio))
                
                transcription_status[job_id]['progress'] = f'Processing chunk {i+1}/{num_chunks}...'
                transcription_status[job_id]['current_chunk'] = i + 1
                
                # Extract chunk
                chunk = audio[start_time:end_time]
                chunk_file = os.path.join(tempfile.gettempdir(), f"chunk_{job_id}_{i}.wav")
                chunk.export(chunk_file, format="wav")
                
                # Transcribe chunk
                try:
                    chunk_result = transcribe_path(chunk_file)
                    chunk_text = chunk_result.get("text", "").strip()
                    if chunk_text:
                        transcribed_chunks.append(chunk_text)
                    
                    # Clean up chunk file
                    if os.path.exists(chunk_file):
                        os.remove(chunk_file)
                        
                except Exception as e:
                    transcribed_chunks.append(f"[Error in chunk {i+1}: {str(e)}]")
            
            # Combine all chunks
            transcribed_text = "\n\n".join(transcribed_chunks)
        
        # Save transcription to file
        output_filename = f"transcription_{job_id}.txt"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(transcribed_text)
        
        # Update status
        transcription_status[job_id]['status'] = 'completed'
        transcription_status[job_id]['progress'] = 'Transcription completed!'
        transcription_status[job_id]['output_file'] = output_filename
        transcription_status[job_id]['text_preview'] = transcribed_text[:500] + "..." if len(transcribed_text) > 500 else transcribed_text
        
        # Update database
        with app.app_context():
            transcription = Transcription.query.filter_by(job_id=job_id).first()
            if transcription:
                transcription.status = 'completed'
                transcription.output_file = output_filename
                transcription.completed_at = datetime.utcnow()
                db.session.commit()
        
        # Clean up temporary files
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
            
    except Exception as e:
        transcription_status[job_id]['status'] = 'error'
        transcription_status[job_id]['progress'] = f'Error: {str(e)}'
        
        # Update database
        with app.app_context():
            transcription = Transcription.query.filter_by(job_id=job_id).first()
            if transcription:
                transcription.status = 'error'
                db.session.commit()

# Admin decorator
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            flash('Admin access required', 'error')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function

def migrate_database():
    """Add missing columns to existing database"""
    try:
        # Check if is_admin column exists
        with app.app_context():
            # Try to query is_admin column
            try:
                db.session.execute(db.text("SELECT is_admin FROM user LIMIT 1"))
                db.session.execute(db.text("SELECT language FROM transcription LIMIT 1"))
                print("✓ Database schema is up to date")
            except Exception:
                try:
                    db.session.execute(db.text("ALTER TABLE user ADD COLUMN is_admin BOOLEAN DEFAULT 0"))
                    db.session.commit()
                except Exception:
                    pass
                try:
                    db.session.execute(db.text("ALTER TABLE transcription ADD COLUMN language VARCHAR(50) DEFAULT 'auto'"))
                    db.session.commit()
                except Exception:
                    pass
                print(" Database migration completed successfully")
    except Exception as e:
        print(f" Database migration error: {e}")

def ensure_default_admin():
    """Ensure the default admin account exists with the expected credentials"""
    with app.app_context():
        user = User.query.filter_by(username='admin').first()
        if user:
            user.is_admin = True
            user.set_password('admin123')
            db.session.commit()
            print("✓ Default admin ensured: username='admin', password reset")
        else:
            user = User(username='admin', email='admin@gmail.com', is_admin=True)
            user.set_password('admin123')
            db.session.add(user)
            db.session.commit()
            print("✓ Default admin created: username='admin', email='admin@gmail.com'")

# Routes
@app.route('/')
def index():
    """Main page - redirect to dashboard if logged in, otherwise show login"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    form = RegistrationForm()
    if form.validate_on_submit():
        # Check if username or email already exists
        if User.query.filter_by(username=form.username.data).first():
            flash('Username already exists. Please choose a different one.', 'error')
            return render_template('register.html', form=form)
        
        if User.query.filter_by(email=form.email.data).first():
            flash('Email already registered. Please use a different email.', 'error')
            return render_template('register.html', form=form)
        
        # Create new user
        user = User(username=form.username.data, email=form.email.data)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    form = LoginForm()
    if form.validate_on_submit():
        identifier = form.username.data.strip()
        # Allow login via username OR email
        user = User.query.filter(db.or_(User.username == identifier, User.email == identifier)).first()
        
        if user and user.check_password(form.password.data):
            login_user(user)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('dashboard'))
        else:
            flash('Invalid credentials', 'error')
    
    return render_template('login.html', form=form)

@app.route('/logout')
@login_required
def logout():
    """User logout"""
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    """User dashboard with transcription history"""
    # Get user's transcriptions, ordered by most recent
    transcriptions = Transcription.query.filter_by(user_id=current_user.id)\
                                       .order_by(Transcription.created_at.desc()).all()
    
    # Get summary stats
    total_transcriptions = len(transcriptions)
    completed_transcriptions = len([t for t in transcriptions if t.status == 'completed'])
    processing_transcriptions = len([t for t in transcriptions if t.status == 'processing'])
    
    return render_template('dashboard.html', 
                         transcriptions=transcriptions,
                         total_transcriptions=total_transcriptions,
                         completed_transcriptions=completed_transcriptions,
                         processing_transcriptions=processing_transcriptions)

@app.route('/upload_page')
@login_required
def upload_page():
    """Upload page (protected)"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    """Handle file upload (protected)"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not supported. Please upload audio files (MP3, WAV, MP4, M4A, FLAC, OGG)'}), 400
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}_{filename}")
    file.save(file_path)
    
    # Get file size
    file_size = os.path.getsize(file_path)
    
    # Set model size back to base (or your previous default)
    model_size = 'base'
    
    # Create database record
    transcription = Transcription(
        job_id=job_id,
        user_id=current_user.id,
        filename=f"{job_id}_{filename}",
        original_filename=file.filename,
        model_size=model_size,
        file_size=file_size
    )
    db.session.add(transcription)
    db.session.commit()
    
    # Initialize transcription status
    transcription_status[job_id] = {
        'status': 'uploaded',
        'filename': file.filename,
        'progress': 'File uploaded, preparing for transcription...',
        'start_time': time.time()
    }
    
    # Start transcription in background thread
    thread = threading.Thread(target=transcribe_audio_file, args=(file_path, job_id, model_size, 8))
    thread.daemon = True
    thread.start()
    
    return jsonify({'job_id': job_id, 'status': 'uploaded'})

@app.route('/status/<job_id>')
@login_required
def check_status(job_id):
    """Check transcription status (protected)"""
    # Verify the job belongs to the current user
    transcription = Transcription.query.filter_by(job_id=job_id, user_id=current_user.id).first()
    if not transcription:
        return jsonify({'error': 'Job not found or access denied'}), 404
    
    if job_id not in transcription_status:
        return jsonify({'error': 'Job not found'}), 404
    
    status = transcription_status[job_id].copy()
    
    # Calculate elapsed time
    if 'start_time' in status:
        elapsed = time.time() - status['start_time']
        status['elapsed_time'] = f"{elapsed:.0f} seconds"
    
    return jsonify(status)

@app.route('/download/<filename>')
@login_required
def download_file(filename):
    """Download transcribed file (protected)"""
    try:
        # Verify the file belongs to the current user
        transcription = Transcription.query.filter_by(output_file=filename, user_id=current_user.id).first()
        if not transcription:
            return "File not found or access denied", 404
        
        file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        if not os.path.exists(file_path):
            return "File not found", 404
        
        return send_file(file_path, as_attachment=True, download_name=f"{transcription.original_filename}_transcription.txt")
    except Exception as e:
        return f"Error downloading file: {str(e)}", 500

@app.route('/results/<job_id>')
@login_required
def results(job_id):
    """Show results page (protected)"""
    # Verify the job belongs to the current user
    transcription = Transcription.query.filter_by(job_id=job_id, user_id=current_user.id).first()
    if not transcription:
        return "Job not found or access denied", 404
    
    if job_id not in transcription_status:
        return "Job not found", 404
    
    status = transcription_status[job_id]
    return render_template('results.html', job_id=job_id, status=status)

@app.route('/delete_transcription/<int:transcription_id>')
@login_required
def delete_transcription(transcription_id):
    """Delete a transcription"""
    transcription = Transcription.query.filter_by(id=transcription_id, user_id=current_user.id).first()
    if not transcription:
        flash('Transcription not found or access denied', 'error')
        return redirect(url_for('dashboard'))
    
    # Delete files
    try:
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], transcription.filename)
        if os.path.exists(upload_path):
            os.remove(upload_path)
        
        if transcription.output_file:
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], transcription.output_file)
            if os.path.exists(output_path):
                os.remove(output_path)
    except Exception as e:
        print(f"Error deleting files: {e}")
    
    # Delete database record
    db.session.delete(transcription)
    db.session.commit()
    
    flash('Transcription deleted successfully', 'success')
    return redirect(url_for('dashboard'))

@app.route('/chat')
@login_required
def chat_page():
    """AI Assistant chat page"""
    return render_template('chat.html')

@app.route('/profile')
@login_required
def profile():
    return render_template('profile.html')

# Admin Routes
@app.route('/admin')
@login_required
@admin_required
def admin_dashboard():
    """Admin dashboard with system overview"""
    # Get system statistics
    total_users = User.query.count()
    total_transcriptions = Transcription.query.count()
    completed_transcriptions = Transcription.query.filter_by(status='completed').count()
    processing_transcriptions = Transcription.query.filter_by(status='processing').count()
    error_transcriptions = Transcription.query.filter_by(status='error').count()
    
    # Recent activity
    recent_users = User.query.order_by(User.created_at.desc()).limit(5).all()
    recent_transcriptions = Transcription.query.order_by(Transcription.created_at.desc()).limit(10).all()
    
    # Calculate total file sizes
    total_file_size = db.session.query(db.func.sum(Transcription.file_size)).scalar() or 0
    
    return render_template('admin/dashboard.html',
                         total_users=total_users,
                         total_transcriptions=total_transcriptions,
                         completed_transcriptions=completed_transcriptions,
                         processing_transcriptions=processing_transcriptions,
                         error_transcriptions=error_transcriptions,
                         recent_users=recent_users,
                         recent_transcriptions=recent_transcriptions,
                         total_file_size=total_file_size)

@app.route('/admin/users')
@login_required
@admin_required
def admin_users():
    """Admin user management page"""
    page = request.args.get('page', 1, type=int)
    search = request.args.get('search', '')
    
    # Build query
    query = User.query
    if search:
        query = query.filter(
            db.or_(
                User.username.contains(search),
                User.email.contains(search)
            )
        )
    
    users = query.order_by(User.created_at.desc()).paginate(
        page=page, per_page=20, error_out=False)
    
    return render_template('admin/users.html', users=users, search=search)

@app.route('/admin/transcriptions')
@login_required
@admin_required
def admin_transcriptions():
    """Admin transcription management page"""
    page = request.args.get('page', 1, type=int)
    status_filter = request.args.get('status', '')
    search = request.args.get('search', '')
    
    # Build query
    query = db.session.query(Transcription).join(User)
    
    if status_filter:
        query = query.filter(Transcription.status == status_filter)
    
    if search:
        query = query.filter(
            db.or_(
                Transcription.original_filename.contains(search),
                User.username.contains(search)
            )
        )
    
    transcriptions = query.order_by(Transcription.created_at.desc()).paginate(
        page=page, per_page=20, error_out=False)
    
    return render_template('admin/transcriptions.html', 
                         transcriptions=transcriptions, 
                         status_filter=status_filter,
                         search=search)

@app.route('/admin/user/<int:user_id>', methods=['GET', 'POST'])
@login_required
@admin_required
def admin_user_detail(user_id):
    """View detailed user information"""
    user = User.query.get_or_404(user_id)
    user_transcriptions = Transcription.query.filter_by(user_id=user_id)\
                                           .order_by(Transcription.created_at.desc()).all()
    
    # User statistics
    total_transcriptions = len(user_transcriptions)
    completed = len([t for t in user_transcriptions if t.status == 'completed'])
    total_file_size = sum([t.file_size or 0 for t in user_transcriptions])
    
    form = ResetPasswordForm()
    if form.validate_on_submit():
        user.set_password(form.password.data)
        db.session.commit()
        flash(f"Password reset for {user.username}", 'success')
        return redirect(url_for('admin_user_detail', user_id=user_id))
    
    return render_template('admin/user_detail.html',
                         user=user,
                         transcriptions=user_transcriptions,
                         total_transcriptions=total_transcriptions,
                         completed_transcriptions=completed,
                         total_file_size=total_file_size,
                         form=form)

@app.route('/admin/reset_password/<int:user_id>', methods=['POST'])
@login_required
@admin_required
def admin_reset_password(user_id):
    """Admin resets a user's password from the users list page"""
    user = User.query.get_or_404(user_id)
    pw = request.form.get('password', '')
    pw2 = request.form.get('password2', '')
    if len(pw) < 6:
        flash('Password must be at least 6 characters', 'error')
        return redirect(url_for('admin_users', search=request.args.get('search', '')))
    if pw != pw2:
        flash('Passwords do not match', 'error')
        return redirect(url_for('admin_users', search=request.args.get('search', '')))
    user.set_password(pw)
    db.session.commit()
    flash(f"Password reset for {user.username}", 'success')
    return redirect(url_for('admin_users', search=request.args.get('search', '')))

@app.route('/admin/reset_password_lookup', methods=['POST'])
@login_required
@admin_required
def admin_reset_password_lookup():
    # Admin resets a user's password by searching username or email
    identifier = request.form.get('identifier', '').strip()
    pw = request.form.get('password', '')
    pw2 = request.form.get('password2', '')
    if not identifier:
        flash('Enter a username or email', 'error')
        return redirect(url_for('admin_users'))
    user = User.query.filter((User.username == identifier) | (User.email == identifier)).first()
    if not user:
        flash('User not found', 'error')
        return redirect(url_for('admin_users'))
    if len(pw) < 6:
        flash('Password must be at least 6 characters', 'error')
        return redirect(url_for('admin_users', search=identifier))
    if pw != pw2:
        flash('Passwords do not match', 'error')
        return redirect(url_for('admin_users', search=identifier))
    user.set_password(pw)
    db.session.commit()
    flash(f"Password reset for {user.username}", 'success')
    return redirect(url_for('admin_users', search=identifier))

@app.route('/admin/toggle_admin/<int:user_id>')
@login_required
@admin_required
def toggle_admin(user_id):
    """Toggle admin status for a user"""
    user = User.query.get_or_404(user_id)
    
    if user.id == current_user.id:
        flash('Cannot modify your own admin status', 'error')
        return redirect(url_for('admin_user_detail', user_id=user_id))
    
    user.is_admin = not user.is_admin
    db.session.commit()
    
    status = 'granted' if user.is_admin else 'revoked'
    flash(f'Admin privileges {status} for {user.username}', 'success')
    
    return redirect(url_for('admin_user_detail', user_id=user_id))

@app.route('/admin/delete_user/<int:user_id>')
@login_required
@admin_required
def admin_delete_user(user_id):
    """Delete a user and all their transcriptions"""
    user = User.query.get_or_404(user_id)
    
    if user.id == current_user.id:
        flash('Cannot delete your own account', 'error')
        return redirect(url_for('admin_users'))
    
    # Delete associated files
    for transcription in user.transcriptions:
        try:
            # Delete uploaded file
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], transcription.filename)
            if os.path.exists(upload_path):
                os.remove(upload_path)
            
            # Delete output file
            if transcription.output_file:
                output_path = os.path.join(app.config['OUTPUT_FOLDER'], transcription.output_file)
                if os.path.exists(output_path):
                    os.remove(output_path)
        except Exception as e:
            print(f"Error deleting files for user {user.username}: {e}")
    
    # Delete user (cascade will delete transcriptions)
    db.session.delete(user)
    db.session.commit()
    
    flash(f'User {user.username} and all associated data deleted successfully', 'success')
    return redirect(url_for('admin_users'))

@app.route('/admin/delete_transcription/<int:transcription_id>', methods=['POST'])
@login_required
@admin_required
def admin_delete_transcription(transcription_id):
    """Delete a specific transcription (POST)"""
    transcription = Transcription.query.get_or_404(transcription_id)
    user_id = transcription.user_id
    # Remove in-memory status entry if present
    try:
        if transcription.job_id in transcription_status:
            transcription_status.pop(transcription.job_id, None)
    except Exception:
        pass
    # Delete files
    try:
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], transcription.filename)
        if os.path.exists(upload_path):
            os.remove(upload_path)
        if transcription.output_file:
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], transcription.output_file)
            if os.path.exists(output_path):
                os.remove(output_path)
    except Exception as e:
        print(f"Error deleting files: {e}")
    # Delete database record
    db.session.delete(transcription)
    db.session.commit()
    flash('Transcription deleted successfully', 'success')
    return redirect(url_for('admin_user_detail', user_id=user_id))

@app.route('/admin/system_stats')
@login_required
@admin_required
def admin_system_stats():
    """System statistics and health"""
    # Syste
    system_info = {
        'platform': platform.system(),
        'platform_version': platform.version(),
        'architecture': platform.architecture()[0],
        'processor': platform.processor(),
        'python_version': platform.python_version(),
    }
    
    # System resources - only if psutil is available
    system_resources = None
    if PSUTIL_AVAILABLE:
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Use different disk path for Windows vs Unix
            disk_path = 'C:\\' if platform.system() == 'Windows' else '/'
            disk = psutil.disk_usage(disk_path)
            
            system_resources = {
                'cpu_percent': cpu_percent,
                'memory_total': memory.total,
                'memory_available': memory.available,
                'memory_percent': memory.percent,
                'disk_total': disk.total,
                'disk_used': disk.used,
                'disk_free': disk.free,
                'disk_percent': (disk.used / disk.total) * 100
            }
        except Exception as e:
            print(f"Error getting system resources: {e}")
            system_resources = None
    
    # Database stats
    db_stats = {
        'total_users': User.query.count(),
        'admin_users': User.query.filter_by(is_admin=True).count(),
        'total_transcriptions': Transcription.query.count(),
        'completed_transcriptions': Transcription.query.filter_by(status='completed').count(),
        'processing_transcriptions': Transcription.query.filter_by(status='processing').count(),
        'error_transcriptions': Transcription.query.filter_by(status='error').count(),
    }
    
    return render_template('admin/system_stats.html',
                         system_info=system_info,
                         system_resources=system_resources,
                         db_stats=db_stats)

@app.route('/admin/create_admin', methods=['GET', 'POST'])
def create_admin():
    """Create initial admin user (only available if no admin exists)"""
    # Check if any admin exists
    admin_exists = User.query.filter_by(is_admin=True).first()
    if admin_exists:
        flash('Admin user already exists', 'error')
        return redirect(url_for('login'))
    
    form = RegistrationForm()
    if form.validate_on_submit():
        # Check if username or email already exists
        if User.query.filter_by(username=form.username.data).first():
            flash('Username already exists. Please choose a different one.', 'error')
            return render_template('admin/create_admin.html', form=form)
        
        # Fix invalid syntax on email check
        if User.query.filter_by(email=form.email.data).first():
            flash('Email already registered. Please use a different email.', 'error')
            return render_template('admin/create_admin.html', form=form)
        
        # Create admin user
        user = User(username=form.username.data, email=form.email.data, is_admin=True)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        
        flash('Admin account created successfully! Please log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('admin/create_admin.html', form=form)

if __name__ == '__main__':
    # Create database tables and handle migrations
    with app.app_context():
        db.create_all()
        migrate_database()
        # Ensure a default admin exists
        ensure_default_admin()
    
    print("Starting Transcription Web Server...")
    print("Access the application at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)



