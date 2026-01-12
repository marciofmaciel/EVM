import os
os.environ["TEMP"] = "F:\\Temp"
os.environ["TMP"] = "F:\\Temp"

import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
from scipy import signal
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import io
from PIL import Image
import tempfile
import os
import uuid
import atexit
st.set_page_config(page_title="EVM - Análise de Tensões Residuais", page_icon="🔬", layout="wide")
# =====================================================
# FUNÇÕES DE PROCESSAMENTO
# =====================================================

def principal_tensor_vectors(heatmap):
    """
    Calcula o maior vetor próprio do tensor de deformação (gradiente) para cada pixel.
    Retorna:
        eigvals: autovalor máximo (H, W)
        eigvecs: vetor próprio correspondente (H, W, 2)
    """
    # Garante float32 para compatibilidade com np.linalg
    heatmap = heatmap.astype(np.float32)
    grad_x = np.gradient(heatmap, axis=1)
    grad_y = np.gradient(heatmap, axis=0)
    H, W = heatmap.shape
    eigvals = np.zeros((H, W), dtype=np.float32)
    eigvecs = np.zeros((H, W, 2), dtype=np.float32)
    for y in range(H):
        for x in range(W):
            # Tensor de deformação simétrico 2x2
            J = np.array([
                [grad_x[y, x], 0.5 * (grad_y[y, x] + grad_x[y, x])],
                [0.5 * (grad_y[y, x] + grad_x[y, x]), grad_y[y, x]]
            ], dtype=np.float32)
            vals, vecs = np.linalg.eigh(J)
            idx = np.argmax(np.abs(vals))
            eigvals[y, x] = vals[idx]
            eigvecs[y, x] = vecs[:, idx]
    return eigvals, eigvecs

def stabilize_video(frames, progress_bar=None):
    """
    Estabiliza sequência de frames usando detecção de features ORB.
    Remove movimento de câmera indesejado.
        Parâmetros:
        - frames: array (T, H, W) de frames em escala de cinza
        - progress_bar: barra de progresso do Streamlit (opcional)
        
        Retorna:
        - frames_stabilized: array estabilizado
    """
    T, H, W = frames.shape
    frames_stabilized = np.zeros_like(frames)
    
    # Frame de referência (primeiro frame)
    frames_stabilized[0] = frames[0]
    reference_frame = frames[0].astype(np.uint8)
    
    # Detector ORB
    orb = cv2.ORB_create(nfeatures=500)
    
    # Detecta keypoints e descritores no frame de referência
    kp_ref, des_ref = orb.detectAndCompute(reference_frame, None)
    
    if des_ref is None or len(kp_ref) < 10:
        st.warning("⚠️ Poucos features detectados. Usando frames originais.")
        return frames
    
    # Matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    for i in range(1, T):
        current_frame = frames[i].astype(np.uint8)
        
        # Detecta keypoints no frame atual
        kp_curr, des_curr = orb.detectAndCompute(current_frame, None)
        
        if des_curr is None or len(kp_curr) < 10:
            # Se falhar, usa frame original
            frames_stabilized[i] = frames[i]
            continue
        
        # Matching
        try:
            matches = bf.match(des_ref, des_curr)
            matches = sorted(matches, key=lambda x: x.distance)
            
            if len(matches) >= 10:
                # Extrai pontos correspondentes
                src_pts = np.float32([kp_ref[m.queryIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_curr[m.trainIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)
                
                # Estima transformação afim
                M, mask = cv2.estimateAffinePartial2D(dst_pts, src_pts)
                
                if M is not None:
                    # Aplica transformação para estabilizar
                    stabilized = cv2.warpAffine(current_frame, M, (W, H))
                    frames_stabilized[i] = stabilized
                else:
                    frames_stabilized[i] = frames[i]
            else:
                frames_stabilized[i] = frames[i]
        except:
            frames_stabilized[i] = frames[i]
        
        # Atualiza barra de progresso
        if progress_bar is not None:
            progress_bar.progress((i + 1) / T)
    
    return frames_stabilized



def read_video(video_path, max_frames=None):
    """
    Lê vídeo e retorna array de frames.
        Parâmetros:
        - video_path: caminho do arquivo de vídeo
        - max_frames: número máximo de frames (None = todos)
        
        Retorna:
        - frames: array (T, H, W, C)
        - fps: taxa de quadros por segundo
    """
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 1000:
        st.warning(f"⚠️ FPS inválido detectado ({fps}). Usando 30 FPS como padrão.")
        fps = 30.0
    
    frames = []
    count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frames.append(frame)
        count += 1
        
        if max_frames is not None and count >= max_frames:
            break
    
    cap.release()
    
    if len(frames) == 0:
        raise ValueError("Não foi possível ler frames do vídeo.")
    
    return np.array(frames), fps



def apply_bandpass_filter(frames_gray, fps, f_low, f_high, order=5, progress_bar=None):
    """
    Aplica filtro passa-banda temporal Butterworth.
        Parâmetros:
        - frames_gray: array (T, H, W) normalizado [0, 1]
        - fps: taxa de quadros
        - f_low: frequência baixa (Hz)
        - f_high: frequência alta (Hz)
        - order: ordem do filtro
        
        Retorna:
        - filtered: array filtrado (T, H, W)
    """
    nyquist = fps / 2.0
    
    if f_high >= nyquist:
        raise ValueError(f"f_high ({f_high} Hz) deve ser menor que a frequência de Nyquist ({nyquist} Hz).")
    
    # Normaliza frequências
    low = f_low / nyquist
    high = f_high / nyquist
    
    # Cria filtro Butterworth
    sos = signal.butter(order, [low, high], btype='band', output='sos')
    
    T, H, W = frames_gray.shape
    filtered = np.zeros_like(frames_gray)

    # Aplica filtro por linha (otimização) e atualiza barra de progresso
    for y in range(H):
        # Processa todos os pixels da linha de uma vez (mais rápido)
        pixel_signals = frames_gray[:, y, :]
        filtered[:, y, :] = signal.sosfiltfilt(sos, pixel_signals, axis=0)
        # Atualiza barra de progresso
        if progress_bar is not None:
            progress_bar.progress((y + 1) / H)
    return filtered



def compute_rms_map(filtered_frames):
    """
    Calcula mapa RMS (Root Mean Square) ao longo do tempo.
        Parâmetros:
        - filtered_frames: array (T, H, W)
        
        Retorna:
        - rms_map: array (H, W)
    """
    # Converte para float16 para economizar memória
    filtered_frames = filtered_frames.astype(np.float16)
    rms_map = np.sqrt(np.mean(filtered_frames ** 2, axis=0))
    return rms_map



def normalize_map(rms_map, p_low=5, p_high=95):
    """
    Normaliza mapa por percentis.
    Retorna array normalizado [0, 1].
    """
    p5 = np.percentile(rms_map, p_low)
    p95 = np.percentile(rms_map, p_high)
    denom = p95 - p5
    # Se p5 == p95 ou denom ~ 0, normaliza pelo valor máximo
    if np.isclose(denom, 0) or not np.isfinite(denom):
        max_val = np.max(rms_map)
        if max_val > 0:
            normalized = rms_map / max_val
        else:
            normalized = np.zeros_like(rms_map)
    else:
        normalized = (rms_map - p5) / denom
        normalized = np.clip(normalized, 0, 1)
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)
    return normalized



def create_heatmap_overlay(frame_bgr, heatmap_normalized, colormap_name='inferno', alpha=0.5):
    """
    Cria overlay do heatmap sobre o frame original.
        Parâmetros:
        - frame_bgr: frame original BGR (H, W, 3)
        - heatmap_normalized: mapa normalizado [0, 1] (H, W)
        - colormap_name: nome do colormap matplotlib
        - alpha: opacidade do overlay [0, 1]
        
        Retorna:
        - overlay: frame com heatmap sobreposto (H, W, 3)
    """
    # Garante que alpha está entre 0 e 1
    alpha = np.clip(alpha, 0, 1)

    # Aplica colormap (compatível com Matplotlib <3.7 e >=3.7)
    cmap = cm.get_cmap(colormap_name)
    heatmap_colored = cmap(heatmap_normalized)[:, :, :3]  # RGB
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

    # Converte para BGR
    heatmap_bgr = cv2.cvtColor(heatmap_colored, cv2.COLOR_RGB2BGR)

    # Garante que ambos são uint8
    if not np.issubdtype(frame_bgr.dtype, np.uint8):
        frame_bgr = np.clip(frame_bgr * 255, 0, 255).astype(np.uint8)
    if not np.issubdtype(heatmap_bgr.dtype, np.uint8):
        heatmap_bgr = np.clip(heatmap_bgr * 255, 0, 255).astype(np.uint8)

    # Blend
    overlay = cv2.addWeighted(frame_bgr, 1 - alpha, heatmap_bgr, alpha, 0)
    return overlay



def write_video(output_path, frames, fps):
    """
    Escreve vídeo a partir de array de frames.
        Parâmetros:
        - output_path: caminho de saída
        - frames: array (T, H, W, 3) BGR
        - fps: taxa de quadros
    """
    T, H, W, C = frames.shape

    # Se output_path for vazio, salva na pasta do projeto
    if not output_path:
        output_path = os.path.join(os.getcwd(), 'output_final.avi')
    else:
        output_path = os.path.abspath(output_path)

    # Tenta MJPG primeiro, depois mp4v
    codecs = [('MJPG', '.avi'), ('mp4v', '.mp4')]
    last_error = None
    for codec, ext in codecs:
        test_path = os.path.splitext(output_path)[0] + ext
        fourcc = cv2.VideoWriter_fourcc(*codec)
        try:
            # Log de debug
            print(f"[DEBUG] Salvando vídeo em {test_path} com codec {codec}")
            print(f"[DEBUG] Frame shape: {frames[0].shape}, dtype: {frames[0].dtype}")
            out = cv2.VideoWriter(test_path, fourcc, fps, (W, H))
            if not out.isOpened():
                raise RuntimeError(f"Não foi possível abrir o arquivo de vídeo para escrita: {test_path}. Verifique permissões, codecs e formato dos frames.")

            for frame in frames:
                # Garante formato uint8 e BGR e que frame é array numpy
                frame_np = np.asarray(frame)
                if frame_np.dtype != np.uint8:
                    frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)
                if frame_np.shape[2] != 3:
                    raise ValueError("Frame deve ter 3 canais (BGR)")
                out.write(frame_np)
            out.release()
            return test_path
        except Exception as e:
            last_error = e
            continue
    raise RuntimeError(f"Erro ao tentar salvar o vídeo. Último erro: {last_error}")



# =====================================================
# INTERFACE STREAMLIT
# =====================================================
st.title("🔬 Análise de Tensões via EVM")
st.markdown("### Eulerian Video Magnification para Resposta Vibracional")
# Aviso crítico
st.error("""
⚠️ AVISO CRÍTICO:
Este aplicativo gera um índice RELATIVO de tensão residual baseado em resposta vibracional.
Os valores NÃO são tensões absolutas (MPa, Pa) e requerem calibração externa e modelos mecânicos
para interpretação quantitativa.
""")
st.info("""
📋 Recomendações para Melhor Resultado:
Use tripé ou suporte fixo (elimina movimento de câmera)
Iluminação constante e uniforme
FPS ≥ 30 para capturar vibrações
Vídeo sem compressão excessiva
Foco fixo (desabilite autofoco)
""")
# =====================================================
# SIDEBAR - PARÂMETROS
# =====================================================
st.sidebar.title("⚙️ Configurações")
# Pré-processamento
st.sidebar.markdown("### 🎥 Pré-processamento")
enable_stabilization = st.sidebar.checkbox(
"Estabilização de vídeo",
value=True,
help="Remove movimento de câmera antes do processamento EVM. Recomendado para vídeos capturados sem tripé."
)
# Parâmetros EVM
st.sidebar.markdown("### 🔧 Parâmetros EVM")
f_low = st.sidebar.number_input(
"Frequência baixa (Hz)",
min_value=0.1,
max_value=100.0,
value=0.5,
step=0.1,
help="Limite inferior da banda passante"
)
f_high = st.sidebar.number_input(
"Frequência alta (Hz)",
min_value=0.1,
max_value=100.0,
value=3.0,
step=0.1,
help="Limite superior da banda passante (deve ser < FPS/2)"
)
alpha = st.sidebar.slider(
"Ganho Alpha",
min_value=1,
max_value=100,
value=20,
help="Fator de amplificação. Valores altos podem causar artefatos."
)
filter_order = st.sidebar.slider(
"Ordem do filtro",
min_value=1,
max_value=10,
value=5,
help="Ordem do filtro Butterworth"
)
st.sidebar.markdown("### 📊 Normalização")
p_low = st.sidebar.slider(
"Percentil baixo",
min_value=0,
max_value=50,
value=5,
help="Remove outliers inferiores"
)
p_high = st.sidebar.slider(
"Percentil alto",
min_value=50,
max_value=100,
value=95,
help="Remove outliers superiores"
)
# Visualização
st.sidebar.markdown("### 🎨 Visualização")
output_mode = st.sidebar.selectbox(
    "Tipo de resultado a gerar",
    options=["Heatmap RMS", "Vídeo com deslocamentos amplificados"],
    index=0,
    help="Escolha entre visualizar o heatmap RMS ou o vídeo com amplificação dos deslocamentos.",
    key="output_mode_selectbox"
)

if output_mode == "Heatmap RMS":
    colormap_name = st.sidebar.selectbox(
        "Colormap",
        options=['inferno', 'turbo', 'viridis', 'plasma', 'jet', 'hot', 'cool'],
        index=0,
        help="Esquema de cores do heatmap",
        key="colormap"
    )
    overlay_alpha = st.sidebar.slider(
        "Opacidade do overlay",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Transparência do heatmap sobre o vídeo original",
        key="overlay_alpha"
    )
    visual_gain = st.sidebar.slider(
        "Ganho Visual do Heatmap (multiplicativo)",
        min_value=0.1,
        max_value=10.0,
        value=1.0,
        step=0.1,
        help="Multiplica o mapa RMS para realçar diferenças visuais.",
        key="visual_gain"
    )

    # Garante que output_frames está definido antes do uso
    output_frames = None

    # LOGS DETALHADOS PARA DEBUG
    # debug_var('heatmap_map', heatmap_map)  # Só pode ser chamado após definição de heatmap_map
    # debug_var('frames_bgr_roi', frames_bgr_roi)  # frames_bgr_roi not defined yet
    # debug_var('filtered_amplified', filtered_amplified)  # Only available after processing
    # debug_var('frames_gray', frames_gray)  # Removed: frames_gray not defined here
    # debug_var('output_frames', output_frames)
    # debug_var('colormap_name', colormap_name)
    # debug_var('overlay_alpha', overlay_alpha)
    # debug_var('visual_gain', visual_gain)
    # Limites de cor do Heatmap REMOVIDOS
    vmin = None
    vmax = None
else:
    colormap_name = None
    overlay_alpha = None
    visual_gain = 1.0
    vmin = None
    vmax = None

# ROI desenhável REMOVIDO
# ...remover todas as variáveis, widgets e lógica relacionadas a ROI, canvas, frame_for_canvas, x0, y0, x1, y1, roi_xmin, roi_xmax, roi_ymin, roi_ymax, use_roi, etc...

# =====================================================
# UPLOAD E PROCESSAMENTO
# =====================================================
st.markdown("---")
st.markdown("## 📤 Upload do Vídeo")
uploaded_file = st.file_uploader(
"Faça upload do vídeo (MP4, AVI)",
type=['mp4', 'avi', 'mov'],
help="Selecione o arquivo de vídeo para análise"
)

# ROI desenhável REMOVIDO (não há mais seleção/desenho de ROI)

# Performance
st.sidebar.markdown("### ⚡ Performance")
max_frames = st.sidebar.number_input(
    "Máximo de frames para preview",
    min_value=10,
    max_value=1000,
    value=100,
    step=10,
    help="Limita processamento para testes rápidos"
)
if uploaded_file is not None:
    # Garante que output_frames está definido antes de qualquer uso
    output_frames = None

    # DEBUG: Log tipos de variáveis críticas
    # def debug_var(name, var):
    #     try:
    #         st.write(f"[DEBUG] {name}: type={type(var)}, shape={getattr(var, 'shape', None)}, value={var if isinstance(var, (int, float, str, bool)) else 'array'}")
    #     except Exception as e:
    #         st.write(f"[DEBUG] {name}: erro ao logar: {e}")

    # output_frames já inicializado acima

    # Exemplo de uso dos logs (pode ser removido ou ajustado conforme necessário)
    # debug_var('output_mode', output_mode)
    # debug_var('output_frames', output_frames)
    # Salva temporariamente
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name
    st.success(f"✅ Vídeo carregado: {uploaded_file.name}")
    
    # Botão de processar
    if st.button("▶️ Processar Vídeo", type="primary"):
        
        try:
            # Leitura do vídeo
            st.info("📹 Lendo vídeo...")
            frames_bgr, fps = read_video(video_path, max_frames=max_frames)
            T, H, W, C = frames_bgr.shape
            
            st.success(f"✅ Vídeo lido: {T} frames, {W}x{H}, {fps:.2f} FPS")
            
            # Validação de Nyquist
            nyquist = fps / 2.0
            if f_high >= nyquist:
                st.error(f"❌ Erro: f_high ({f_high} Hz) deve ser menor que FPS/2 ({nyquist:.2f} Hz).")
                st.stop()
            
            # Redimensionamento automático para economizar memória
            target_width, target_height = 640, 360
            H0, W0 = frames_bgr[0].shape[:2]
            if W0 > target_width or H0 > target_height:
                st.warning(f"🔄 Redimensionando frames para {target_width}x{target_height} para evitar erro de memória.")
                frames_bgr = np.array([
                    cv2.resize(f, (target_width, target_height), interpolation=cv2.INTER_AREA)
                    for f in frames_bgr
                ])
            # Conversão para escala de cinza
            st.info("🎨 Convertendo para escala de cinza...")
            frames_gray = np.array([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames_bgr])
            frames_gray = frames_gray.astype(np.float32) / 255.0

            # ROI: recorte dos frames REMOVIDO

            frames_bgr_roi = frames_bgr  # sempre igual ao original
            
            # Estabilização (opcional)
            if enable_stabilization:
                st.info("🎥 Estabilizando vídeo...")
                st.write("[LOG] Iniciando estabilização dos frames...")
                stab_progress = st.progress(0)
                frames_gray = stabilize_video(frames_gray, stab_progress)
                st.success("✅ Vídeo estabilizado!")
                st.write("[LOG] Estabilização concluída.")
            
            # Aplicação do filtro EVM
            st.info(f"🔧 Aplicando filtro passa-banda [{f_low}-{f_high} Hz]...")
            st.write(f"[LOG] Filtro Butterworth: ordem={filter_order}, f_low={f_low}, f_high={f_high}, fps={fps}")
            progress_bar = st.progress(0)
            filtered = apply_bandpass_filter(frames_gray, fps, f_low, f_high, filter_order, progress_bar)
            progress_bar.progress(1.0)
            st.write("[LOG] Filtro passa-banda aplicado.")

            # Aplica Ganho Alpha diretamente ao sinal filtrado
            st.info(f"🔊 Aplicando Ganho Alpha = {alpha} ao sinal filtrado...")
            st.write(f"[LOG] Multiplicando sinal filtrado por alpha={alpha}")
            filtered_amplified = filtered * alpha

            # Cálculo do mapa RMS
            st.info("📊 Calculando mapa RMS...")
            st.write("[LOG] Calculando RMS dos frames amplificados...")
            rms_map = compute_rms_map(filtered_amplified)
            st.write(f"[LOG] RMS map (com ganho) - min: {np.min(rms_map):.6f}, max: {np.max(rms_map):.6f}, mean: {np.mean(rms_map):.6f}, std: {np.std(rms_map):.6f}")
            progress_bar.progress(0.7)

            if output_mode == "Heatmap RMS":
                st.info("🟢 Gerando heatmap RMS absoluto de toda a imagem...")
                st.write(f"[LOG] Usando mapa RMS absoluto com visual_gain={visual_gain}")
                heatmap_map = rms_map * visual_gain
                st.write(f"[LOG] Heatmap RMS absoluto: min={np.min(heatmap_map):.6f}, max={np.max(heatmap_map):.6f}, mean={np.mean(heatmap_map):.6f}, std={np.std(heatmap_map):.6f}")
                progress_bar.progress(0.9)
                # debug_var('heatmap_map', heatmap_map)
            else:
                st.info("🎬 Gerando vídeo com deslocamentos amplificados...")
                heatmap_map = None
                progress_bar.progress(0.9)
            
            # Geração do vídeo de saída
            st.info("🎬 Gerando vídeo de saída com overlay...")
            output_frames = []

            if output_mode == "Heatmap RMS":
                # Calcula os tensores principais uma vez para todos os frames
                eigvals, eigvecs = principal_tensor_vectors(heatmap_map)
                threshold = np.percentile(np.abs(eigvals), 99)  # 1% mais significativos
                significant_idxs = np.argwhere(np.abs(eigvals) >= threshold)
                max_vectors = 30
                if significant_idxs.shape[0] > max_vectors:
                    rng = np.random.default_rng(seed=42)
                    selected = rng.choice(significant_idxs.shape[0], size=max_vectors, replace=False)
                    significant_idxs = significant_idxs[selected]

                H, W = heatmap_map.shape
                diag = np.sqrt(H**2 + W**2)
                eigvals_abs = np.abs(eigvals)
                if len(significant_idxs) > 0:
                    min_val = np.min(eigvals_abs[tuple(significant_idxs.T)])
                    max_val = np.max(eigvals_abs[tuple(significant_idxs.T)])
                else:
                    min_val = 0
                    max_val = 1
                # Índice do tensor mais crítico
                if len(significant_idxs) > 0:
                    idx_crit = np.argmax(eigvals_abs[tuple(significant_idxs.T)])
                    y_crit, x_crit = significant_idxs[idx_crit]
                else:
                    y_crit, x_crit = None, None

                # Cria máscara para amplificar apenas os 30 tensores máximos
                mask_amplify = np.zeros((H, W), dtype=np.float32)
                for y, x in significant_idxs:
                    mask_amplify[y, x] = 1.0
                # Suaviza a máscara para evitar artefatos
                mask_amplify = cv2.GaussianBlur(mask_amplify, (7, 7), 0)

                # Frame base congelado (primeiro frame em cinza)
                frozen_gray = cv2.cvtColor(frames_bgr_roi[0], cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

                for i, frame_bgr in enumerate(frames_bgr_roi):
                    if filtered.shape[1:] == frame_bgr.shape[:2]:
                        # Amplifica apenas nos locais dos 30 tensores máximos, mantendo o movimento do vídeo
                        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
                        amplified_gray = frame_gray + (filtered[i] * mask_amplify * alpha)
                        amplified_gray = np.clip(amplified_gray, 0, 1)
                        amplified_bgr = cv2.cvtColor((amplified_gray * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                    else:
                        amplified_bgr = frame_bgr

                    # Overlay do heatmap
                    overlay = create_heatmap_overlay(
                        amplified_bgr,
                        np.abs(eigvals) / (np.max(np.abs(eigvals)) + 1e-8),
                        colormap_name=colormap_name,
                        alpha=overlay_alpha
                    )
                    overlay_vec = overlay.copy()

                    # Desenha os 30 tensores principais no frame
                    for j, (y, x) in enumerate(significant_idxs):
                        v = eigvecs[y, x]
                        norm = np.linalg.norm(v)
                        if norm == 0:
                            continue
                        scale = 0.1 * diag
                        x1 = int(round(x + v[0] * scale))
                        y1 = int(round(y + v[1] * scale))
                        val = eigvals_abs[y, x]
                        t = (val - min_val) / (max_val - min_val) if max_val > min_val else 0
                        hue = int(120 * (1 - t))  # 120=azul, 0=vermelho
                        sat = int(200 + 55 * t)
                        val_cv = int(200 + 55 * t)
                        if y == y_crit and x == x_crit:
                            hue = 0
                            sat = 255
                            val_cv = 255
                        hsv_color = np.uint8([[[hue, sat, val_cv]]])
                        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0, 0].tolist()
                        color = tuple(int(c) for c in bgr_color)
                        cv2.arrowedLine(
                            overlay_vec,
                            (x, y),
                            (x1, y1),
                            color,
                            thickness=2,
                            tipLength=0.2
                        )
                    output_frames.append(overlay_vec)
                output_frames = np.array(output_frames)
                progress_bar.progress(1.0)
                st.success("✅ Processamento concluído!")
            else:
                # EVM Laplaciano em tons de cinza com transições suaves (melhor qualidade visual)
                st.info("🔬 Aplicando EVM Laplaciano em tons de cinza (transições suaves, sem overlay)...")
                n_levels = 4  # Níveis da pirâmide Laplaciana
                output_frames = []
                T = frames_bgr_roi.shape[0]
                H, W = frames_bgr_roi.shape[1:3]

                # Converte frames para escala de cinza [0,1]
                frames_gray = np.array([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames_bgr_roi]).astype(np.float32) / 255.0

                # 1. Construir pirâmide Laplaciana para todos os frames (em cinza)
                laplacian_pyrs = []
                for i in range(T):
                    frame = frames_gray[i]
                    pyr = []
                    current = frame
                    for _ in range(n_levels):
                        next_ = cv2.pyrDown(current)
                        up = cv2.pyrUp(next_, dstsize=(current.shape[1], current.shape[0]))
                        lap = current - up
                        pyr.append(lap)
                        current = next_
                    pyr.append(current)  # nível mais baixo
                    laplacian_pyrs.append(pyr)

                # 2. Para cada nível da pirâmide (exceto o mais baixo), empilhar no tempo e aplicar filtro temporal
                filtered_pyrs = []
                for level in range(n_levels):
                    # Empilha todos os frames desse nível: shape (T, H, W)
                    level_stack = np.stack([laplacian_pyrs[t][level] for t in range(T)], axis=0)
                    # Aplica filtro temporal passa-banda com suavização adicional (janela de média móvel)
                    filtered_level = apply_bandpass_filter(level_stack, fps, f_low, f_high, filter_order)
                    # Suavização temporal extra para transições suaves
                    kernel_size = 5  # tamanho da janela (ímpar)
                    if kernel_size > 1:
                        pad = kernel_size // 2
                        filtered_level = np.pad(filtered_level, ((pad, pad), (0, 0), (0, 0)), mode='edge')
                        filtered_level = np.array([
                            np.mean(filtered_level[i:i+kernel_size], axis=0)
                            for i in range(T)
                        ])
                    # Amplifica
                    filtered_level *= alpha
                    filtered_pyrs.append(filtered_level)
                # O nível mais baixo não é amplificado
                lowpass_stack = np.stack([laplacian_pyrs[t][-1] for t in range(T)], axis=0)

                # 3. Reconstruir frames amplificados
                for t in range(T):
                    # Começa pelo nível mais baixo (sem amplificação)
                    recon = lowpass_stack[t]
                    for level in reversed(range(n_levels)):
                        recon = cv2.pyrUp(recon, dstsize=filtered_pyrs[level][t].shape[::-1])
                        recon = recon + filtered_pyrs[level][t]
                    recon = np.clip(recon, 0, 1)

                    # --- Mantém tons de cinza originais, mas destaca apenas os tensores de deslocamento em vermelho ---
                    disp_norm = np.abs(recon)
                    disp_norm = disp_norm / (disp_norm.max() + 1e-8)
                    threshold = np.percentile(disp_norm, 90)
                    mask = disp_norm >= threshold

                    # Luminância original normalizada
                    luminance = frames_gray[t]
                    luminance = np.clip(luminance, 0, 1)

                    # Cria imagem BGR em tons de cinza do frame original
                    gray_bgr = cv2.cvtColor((luminance * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

                    # Destaca apenas os tensores de deslocamento (pixels da máscara) em vermelho puro
                    recon_bgr = gray_bgr.copy()
                    recon_bgr[mask] = [0, 0, 255]  # BGR: vermelho puro

                    output_frames.append(recon_bgr)
                output_frames = np.array(output_frames)
                progress_bar.progress(1.0)
                st.success("✅ Processamento concluído!")
        except Exception as e:
            st.error(f"❌ Erro durante o processamento do vídeo: {e}")
# =====================================================
# VISUALIZAÇÃO DOS RESULTADOS
# =====================================================
st.markdown("---")
st.markdown("## 📊 Resultados")

# Só mostra resultados se as variáveis existem e foram processadas
if (
    uploaded_file is not None
    and 'output_frames' in locals() and output_frames is not None
    and isinstance(output_frames, np.ndarray)
    and output_frames.shape[0] > 0
):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🖼️ Frame Original")
        st.image(cv2.cvtColor(output_frames[0], cv2.COLOR_BGR2RGB), use_container_width=True)
    with col2:
        if output_mode == "Heatmap RMS" and 'heatmap_map' in locals() and heatmap_map is not None:
            st.markdown("### 🔥 Heatmap de Tensão (RMS)")
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(heatmap_map, cmap=colormap_name)
            ax.set_title("Mapa RMS de Tensão")
            ax.axis('off')
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Valor RMS', rotation=270, labelpad=20)
            st.pyplot(fig)
        else:
            st.markdown("### 🎬 Vídeo com Deslocamentos Amplificados")
            st.info("O vídeo gerado apresenta os deslocamentos amplificados, sem overlay de heatmap.")
            # Removido: exibição do vídeo na tela principal
            # (O vídeo estará disponível apenas para download na barra lateral)

    # Permitir download dos arquivos gerados na barra lateral
    with st.sidebar:
        st.markdown("### ⬇️ Downloads dos Arquivos Gerados")
        if output_mode == "Heatmap RMS" and 'heatmap_map' in locals() and heatmap_map is not None:
            # Salva heatmap como imagem PNG em buffer
            heatmap_img = (normalize_map(heatmap_map, p_low, p_high) * 255).astype(np.uint8)
            heatmap_img_color = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_INFERNO)
            is_success, buffer = cv2.imencode(".png", heatmap_img_color)
            if is_success:
                st.download_button(
                    label="📥 Baixar Heatmap RMS (PNG)",
                    data=buffer.tobytes(),
                    file_name="heatmap_rms.png",
                    mime="image/png"
                )
            # Salva heatmap como CSV
            csv_buffer = io.StringIO()
            pd.DataFrame(heatmap_map).to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Baixar Heatmap RMS (CSV)",
                data=csv_buffer.getvalue(),
                file_name="heatmap_rms.csv",
                mime="text/csv"
            )
        # Download do vídeo de saída (com overlay ou amplificado)
        if output_frames is not None and isinstance(output_frames, np.ndarray) and output_frames.shape[0] > 0:
            # Salva o vídeo gerado em arquivo temporário
            video_path_out = write_video(None, output_frames, fps)
            with open(video_path_out, "rb") as f:
                st.download_button(
                    label="📥 Baixar Vídeo Gerado",
                    data=f,
                    file_name="video_resultado.mp4",
                    mime="video/mp4"
                )