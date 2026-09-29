import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from datetime import datetime
import os
import json
import warnings
import threading

warnings.filterwarnings('ignore')


class Config:
    def __init__(self, config_file="base.json"):
        self.config_file = config_file
        self.data = self._load()

    def _load(self):
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        raise FileNotFoundError(f"Файл конфигурации {self.config_file} не найден")

    def get(self, key, default=None):
        keys = key.split('.')
        value = self.data
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value

    def get_label(self, key, **kwargs):
        template = self.get(f'labels.{key}', '')
        if kwargs:
            return template.format(**kwargs)
        return template

    def get_phase_label(self, phase):
        return self.get(f'tables.table1.labels.{phase}', phase)

    def get_annual_label(self, state):
        return self.get(f'tables.table2.labels.{state}', state)

    def get_seasonal_label(self, phase):
        return self.get(f'tables.table3.labels.{phase}', phase)

    def get_diurnal_label(self, phase):
        return self.get(f'tables.table4.labels.{phase}', phase)

    def get_anomaly_label(self, atype):
        return self.get(f'tables.table5.labels.{atype}', atype)

    def get_extremum_label(self, extremum):
        return self.get(f'tables.table6.labels.{extremum}', extremum)

    def get_table_rules(self, table_name):
        return self.get(f'tables.{table_name}.rules', [])

    def get_table_title(self, table_name):
        return self.get(f'tables.{table_name}.title', '')

    def save(self):
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=4)


class BinaryDataReader:
    def __init__(self, cache_dir=None, config=None):
        self.config = config or Config()
        if cache_dir is None:
            self.cache_dir = self.config.get('cache.default_path')
        else:
            self.cache_dir = cache_dir
        self.years_index = self._load_years_index()

    def _load_years_index(self):
        index_file = os.path.join(self.cache_dir, 'years_index.json')
        if not os.path.exists(index_file):
            return None
        with open(index_file, 'r') as f:
            return json.load(f)

    def get_available_years(self):
        if self.years_index:
            return sorted([int(y) for y in self.years_index.keys()])
        return []

    def load_data(self, selected_years=None):
        if selected_years is None:
            selected_years = self.get_available_years()

        if not selected_years:
            return [], np.array([])

        all_timestamps = []
        all_values = []

        for year in selected_years:
            year_file = os.path.join(self.cache_dir, f'year_{year}.bin')
            if os.path.exists(year_file):
                year_data = np.fromfile(year_file, dtype=np.float64)
                year_data = year_data.reshape(-1, 2)
                all_timestamps.append(year_data[:, 0])
                all_values.append(year_data[:, 1])

        if not all_timestamps:
            return [], np.array([])

        timestamps = np.concatenate(all_timestamps)
        values = np.concatenate(all_values)
        sort_idx = np.argsort(timestamps)
        timestamps = timestamps[sort_idx]
        values = values[sort_idx]

        dates = [datetime.fromtimestamp(ts) for ts in timestamps]
        return dates, values


class SpectralAnalyzer:
    @staticmethod
    def decompose(signal, dates, data_time_step=15):
        T = len(signal)
        if T % 2 != 0:
            T = T - 1
            signal = signal[:T]
            dates = dates[:T]

        points_per_day = 24 * 60 // data_time_step

        f_hat = np.fft.rfft(signal)
        dt_days = 1.0 / points_per_day
        freqs = np.fft.rfftfreq(T, d=dt_days)

        g_11yr = np.exp(-freqs ** 2 / (2 * (1.0 / (5 * 365) * 0.3) ** 2))
        g_year = np.exp(-(freqs - 1.0 / 365) ** 2 / (2 * (1.0 / 365 * 0.15) ** 2)) * (1.0 - g_11yr)
        g_season = np.exp(-(freqs - 1.0 / 90) ** 2 / (2 * (1.0 / 90 * 0.2) ** 2))
        g_season = g_season * (1.0 - g_11yr) * (1.0 - g_year)
        g_day = np.exp(-(freqs - 1.0) ** 2 / (2 * 1.0 ** 2))
        g_day = g_day * (1.0 - g_11yr) * (1.0 - g_year) * (1.0 - g_season)

        cutoff = 2.0
        steepness = 5.0
        g_hf = 1.0 / (1.0 + np.exp(-steepness * (freqs - cutoff)))
        g_hf = g_hf * (1.0 - g_11yr) * (1.0 - g_year) * (1.0 - g_season) * (1.0 - g_day)

        components = {
            '11yr': np.fft.irfft(f_hat * g_11yr, n=T),
            'annual': np.fft.irfft(f_hat * g_year, n=T),
            'seasonal': np.fft.irfft(f_hat * g_season, n=T),
            'diurnal': np.fft.irfft(f_hat * g_day, n=T),
            'hf': np.fft.irfft(f_hat * g_hf, n=T)
        }

        return components, T, points_per_day, dates


class ExpertSystemCore:
    def __init__(self, config=None):
        self.config = config or Config()
        self.percentiles = {}
        self.component_data = {}
        self.phase_history = []

        for comp in ['original', '11yr', 'annual', 'seasonal', 'diurnal']:
            self.percentiles[comp] = {'p10': None, 'p25': None, 'p50': None, 'p75': None, 'p90': None}

    def calculate_percentiles(self, data, component_name):
        no_data_value = self.config.get('anomaly_thresholds.no_data_value', 9999)
        valid_data = data[data != no_data_value]
        if len(valid_data) > 0:
            self.percentiles[component_name]['p10'] = np.percentile(valid_data, 10)
            self.percentiles[component_name]['p25'] = np.percentile(valid_data, 25)
            self.percentiles[component_name]['p50'] = np.percentile(valid_data, 50)
            self.percentiles[component_name]['p75'] = np.percentile(valid_data, 75)
            self.percentiles[component_name]['p90'] = np.percentile(valid_data, 90)

    def calculate_all_percentiles(self, original, comp_11yr, comp_annual, comp_seasonal, comp_diurnal):
        self.component_data['original'] = original
        self.component_data['11yr'] = comp_11yr
        self.component_data['annual'] = comp_annual
        self.component_data['seasonal'] = comp_seasonal
        self.component_data['diurnal'] = comp_diurnal

        self.calculate_percentiles(original, 'original')
        self.calculate_percentiles(comp_11yr, '11yr')
        self.calculate_percentiles(comp_annual, 'annual')
        self.calculate_percentiles(comp_seasonal, 'seasonal')
        self.calculate_percentiles(comp_diurnal, 'diurnal')

    def analyze_solar_phase(self, value, index, prev_phase=None, prev_phase_conf=1.0, component='11yr'):
        p10 = self.percentiles[component]['p10']
        p90 = self.percentiles[component]['p90']

        comp_full = self.component_data.get(component, [])
        if len(comp_full) == 0:
            return 'UNKNOWN', 0.5, self.config.get_label('no_data_component')

        if p10 is None or p90 is None:
            return 'UNKNOWN', 0.5, self.config.get_label('no_percentiles')

        if value > p90:
            return 'MAXIMUM', 0.85, self.config.get_label('solar_maximum', value=value, p90=p90)

        if value < p10:
            return 'MINIMUM', 0.85, self.config.get_label('solar_minimum', value=value, p10=p10)

        if prev_phase == 'MINIMUM' and value >= p10:
            total_conf = 0.85 * 0.8
            return 'GROWTH', total_conf, self.config.get_label('solar_growth', val=value)

        if prev_phase == 'GROWTH' and value <= p90:
            return 'GROWTH', prev_phase_conf, self.config.get_label('continuation_growth')

        if prev_phase == 'MAXIMUM' and value <= p90:
            total_conf = 0.85 * 0.8
            return 'DECLINE', total_conf, self.config.get_label('solar_decline', val=value)

        if prev_phase == 'DECLINE' and value >= p10:
            return 'DECLINE', prev_phase_conf, self.config.get_label('continuation_decline')

        last_extremum = None
        last_extremum_index = -1
        last_extremum_value = None

        for i in range(index - 1, -1, -1):
            val = comp_full[i]
            if val < p10:
                last_extremum = 'MINIMUM'
                last_extremum_index = i
                last_extremum_value = val
                break
            elif val > p90:
                last_extremum = 'MAXIMUM'
                last_extremum_index = i
                last_extremum_value = val
                break

        if last_extremum is not None:
            has_other_extremum = False
            for i in range(last_extremum_index + 1, index):
                if comp_full[i] < p10 or comp_full[i] > p90:
                    has_other_extremum = True
                    break

            if not has_other_extremum:
                if last_extremum == 'MINIMUM' and value >= p10:
                    total_conf = 0.85 * 0.8
                    return 'GROWTH', total_conf, self.config.get_label('solar_growth', val=last_extremum_value)
                if last_extremum == 'MAXIMUM' and value <= p90:
                    total_conf = 0.85 * 0.8
                    return 'DECLINE', total_conf, self.config.get_label('solar_decline', val=last_extremum_value)

        return 'TRANSITION', 0.6, self.config.get_label('solar_transition', value=value, p10=p10, p90=p90)

    def analyze_annual_anomaly(self, value, month):
        p25 = self.percentiles['annual']['p25']
        p75 = self.percentiles['annual']['p75']
        winter = self.config.get('months.winter', [12, 1, 2])
        summer = self.config.get('months.summer', [6, 7, 8])

        if p25 is None or p75 is None:
            return 'NORMAL', 0.5, self.config.get_label('no_percentiles')

        if month in winter and value > p75:
            return 'WINTER_ANOMALY', 0.9, self.config.get_label('winter_anomaly', value=value, p75=p75)

        if month in summer and value < p25:
            return 'SUMMER_MINIMUM', 0.85, self.config.get_label('summer_minimum', value=value, p25=p25)

        return 'NORMAL', 0.7, self.config.get_label('seasonal_normal')

    def analyze_seasonal_cycle(self, value, month):
        p25 = self.percentiles['seasonal']['p25']
        p75 = self.percentiles['seasonal']['p75']
        spring = self.config.get('months.spring_months', [3, 4])
        autumn = self.config.get('months.autumn_months', [9, 10])
        summer = self.config.get('months.summer', [6, 7, 8])
        winter = self.config.get('months.winter', [12, 1, 2])

        if p25 is None or p75 is None:
            return 'NORMAL', 0.5, self.config.get_label('no_percentiles')

        if month in spring and value > p75:
            return 'SPRING_MAXIMUM', 0.8, self.config.get_label('spring_maximum', value=value, p75=p75)

        if month in autumn and value > p75:
            return 'AUTUMN_MAXIMUM', 0.8, self.config.get_label('autumn_maximum', value=value, p75=p75)

        if month in summer and value < p25:
            return 'SUMMER_MINIMUM', 0.8, self.config.get_label('summer_minimum_seasonal', value=value, p25=p25)

        if month in winter and value < p25:
            return 'WINTER_MINIMUM', 0.75, self.config.get_label('winter_minimum', value=value, p25=p25)

        return 'NORMAL', 0.6, self.config.get_label('diurnal_normal')

    def analyze_diurnal_cycle(self, value, hour):
        p50 = self.percentiles['diurnal']['p50']
        day = self.config.get('time_ranges.day_maximum', {'start': 12, 'end': 15})
        morning = self.config.get('time_ranges.morning_minimum', {'start': 3, 'end': 6})

        if p50 is None:
            return 'NORMAL', 0.5, self.config.get_label('no_percentiles')

        if day['start'] <= hour <= day['end'] and value > p50 * 1.2:
            return 'DAY_MAXIMUM', 0.9, self.config.get_label('day_maximum', value=value, threshold=p50 * 1.2)

        if morning['start'] <= hour <= morning['end'] and value < p50 * 0.8:
            return 'MORNING_MINIMUM', 0.9, self.config.get_label('morning_minimum', value=value, threshold=p50 * 0.8)

        return 'NORMAL', 0.7, self.config.get_label('diurnal_normal')

    def detect_anomaly(self, value, prev_value=None, next_value=None):
        no_data = self.config.get('anomaly_thresholds.no_data_value', 9999)
        max_jump = self.config.get('anomaly_thresholds.max_jump', 5.0)

        if value == no_data:
            return 'NO_DATA', 0.99, self.config.get_label('no_data')

        if prev_value is not None and next_value is not None:
            if abs(value - prev_value) > max_jump and abs(value - next_value) > max_jump and abs(prev_value - next_value) < 1:
                return 'MEASUREMENT_ERROR', 0.9, self.config.get_label('measurement_error')

        return 'NORMAL', 0.95, self.config.get_label('no_anomaly')

    def predict_extremum(self, current_phase):
        if current_phase == 'GROWTH':
            return 'MAXIMUM', 2.5, 0.7, self.config.get_label('prediction_maximum')
        elif current_phase == 'MAXIMUM':
            return 'MINIMUM', 5.5, 0.95, self.config.get_label('prediction_minimum_from_max')
        elif current_phase == 'DECLINE':
            return 'MINIMUM', 2.5, 0.7, self.config.get_label('prediction_minimum')
        elif current_phase == 'MINIMUM':
            return 'MAXIMUM', 5.5, 0.95, self.config.get_label('prediction_maximum_from_min')

        return None, None, 0, self.config.get_label('no_prediction')


class IonosphereExpertSystem:
    def __init__(self):
        self.root = tk.Tk()
        self.config = Config()
        self.root.title(self.config.get_label('app_title'))
        self.root.geometry("1200x800")

        self.reader = BinaryDataReader(config=self.config)
        self.analyzer = SpectralAnalyzer()
        self.es_core = ExpertSystemCore(config=self.config)

        self.dates = []
        self.values = []
        self.components = None
        self.selected_years = []
        self.selected_point_idx = 0

        self.create_widgets()
        self.show_main_page()

    def create_widgets(self):
        self.main_frame = ttk.Frame(self.root, padding=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

    def clear_frame(self):
        for widget in self.main_frame.winfo_children():
            widget.destroy()

    def show_main_page(self):
        self.clear_frame()

        station_name = self.config.get('station.name', 'BC840')

        ttk.Label(self.main_frame, text=self.config.get_label('app_title'),
                  font=("Arial", 18, "bold")).pack(pady=10)
        ttk.Label(self.main_frame, text=f"{self.config.get_label('app_subtitle')} - {station_name}",
                  font=("Arial", 12)).pack()

        info_frame = ttk.LabelFrame(self.main_frame, text=self.config.get_label('info_title'), padding=10)
        info_frame.pack(fill=tk.X, pady=10, padx=20)

        available_years = self.reader.get_available_years()
        if available_years:
            ttk.Label(info_frame, text=f"{self.config.get_label('years_available')} {available_years}",
                      foreground="green").pack(anchor=tk.W)
            if self.selected_years:
                ttk.Label(info_frame, text=f"{self.config.get_label('years_loaded')} {self.selected_years}",
                          foreground="blue").pack(anchor=tk.W)
                ttk.Label(info_frame, text=f"{self.config.get_label('points_count')} {len(self.dates)}",
                          foreground="blue").pack(anchor=tk.W)
                if self.components is None and len(self.dates) > 0:
                    ttk.Label(info_frame, text="⚠ Не выполнено спектральное разложение. Нажмите кнопку 2.",
                              foreground="orange").pack(anchor=tk.W, pady=5)
                elif self.components is not None:
                    ttk.Label(info_frame, text="✓ Спектральное разложение выполнено",
                              foreground="green").pack(anchor=tk.W, pady=5)
        else:
            ttk.Label(info_frame, text=self.config.get_label('cache_not_found'),
                      foreground="orange").pack(anchor=tk.W)

        btn_frame = ttk.Frame(self.main_frame)
        btn_frame.pack(pady=20)

        ttk.Button(btn_frame, text=self.config.get_label('btn_load_data'),
                   command=self.load_data, width=50).pack(pady=5)
        ttk.Button(btn_frame, text=self.config.get_label('btn_spectral'),
                   command=self.run_spectral_analysis, width=50).pack(pady=5)
        ttk.Button(btn_frame, text=self.config.get_label('btn_results'),
                   command=self.show_results, width=50).pack(pady=5)
        ttk.Button(btn_frame, text=self.config.get_label('btn_knowledge'),
                   command=self.show_knowledge_base, width=50).pack(pady=5)
        ttk.Button(btn_frame, text=self.config.get_label('btn_cache_config'),
                   command=self.configure_cache, width=50).pack(pady=5)

    def configure_cache(self):
        dialog = tk.Toplevel(self.root)
        dialog.title(self.config.get_label('title_cache_config'))
        dialog.geometry("500x150")
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(dialog, text=self.config.get_label('label_cache_path'), font=("Arial", 10)).pack(pady=10)

        current_path = self.config.get('cache.default_path')
        path_var = tk.StringVar(value=current_path)
        entry = ttk.Entry(dialog, textvariable=path_var, width=60)
        entry.pack(pady=10)

        def save_path():
            self.config.data['cache']['default_path'] = path_var.get()
            self.config.save()
            self.reader = BinaryDataReader(cache_dir=path_var.get(), config=self.config)
            dialog.destroy()
            messagebox.showinfo(self.config.get_label('success'), self.config.get_label('cache_updated'))
            self.show_main_page()

        ttk.Button(dialog, text=self.config.get_label('btn_save'), command=save_path).pack(pady=10)

    def load_data(self):
        available_years = self.reader.get_available_years()

        if not available_years:
            messagebox.showerror(self.config.get_label('error'), self.config.get_label('no_years_error'))
            return

        dialog = tk.Toplevel(self.root)
        dialog.title(self.config.get_label('title_select_years'))
        dialog.geometry("450x400")
        dialog.transient(self.root)
        dialog.grab_set()

        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (450 // 2)
        y = (dialog.winfo_screenheight() // 2) - (400 // 2)
        dialog.geometry(f"450x400+{x}+{y}")

        ttk.Label(dialog, text=self.config.get_label('label_select_years'), font=("Arial", 11)).pack(pady=10)

        list_frame = ttk.Frame(dialog)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=5)

        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        listbox = tk.Listbox(list_frame, selectmode=tk.MULTIPLE, yscrollcommand=scrollbar.set, height=10)
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=listbox.yview)

        for year in available_years:
            listbox.insert(tk.END, year)

        select_frame = ttk.Frame(dialog)
        select_frame.pack(pady=5)

        ttk.Button(select_frame, text=self.config.get_label('btn_select_all'),
                   command=lambda: listbox.select_set(0, tk.END)).pack(side=tk.LEFT, padx=5)
        ttk.Button(select_frame, text=self.config.get_label('btn_clear'),
                   command=lambda: listbox.selection_clear(0, tk.END)).pack(side=tk.LEFT, padx=5)

        ttk.Separator(dialog, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=20, pady=10)

        action_frame = ttk.Frame(dialog)
        action_frame.pack(pady=15)

        def do_load():
            selected = listbox.curselection()
            if selected:
                self.selected_years = [int(listbox.get(i)) for i in selected]
            else:
                self.selected_years = available_years

            self.dates, self.values = self.reader.load_data(self.selected_years)

            if len(self.dates) == 0:
                messagebox.showerror(self.config.get_label('error'), self.config.get_label('no_data_error'))
                dialog.destroy()
                return

            dialog.destroy()

            msg = self.config.get_label('load_success',
                                        points=len(self.dates),
                                        years=self.selected_years,
                                        start=self.dates[0].date(),
                                        end=self.dates[-1].date())
            messagebox.showinfo(self.config.get_label('success'), msg)
            self.show_main_page()

        ttk.Button(action_frame, text=self.config.get_label('btn_load'), command=do_load, width=12).pack(side=tk.LEFT, padx=10)
        ttk.Button(action_frame, text=self.config.get_label('btn_cancel'), command=dialog.destroy, width=12).pack(side=tk.LEFT, padx=10)

    def run_spectral_analysis(self):
        if len(self.dates) == 0:
            messagebox.showwarning(self.config.get_label('warning'), self.config.get_label('no_data_warning'))
            return

        def analyze():
            if len(self.dates) > 1:
                diffs = [(self.dates[i] - self.dates[i - 1]).total_seconds() / 60
                         for i in range(1, min(100, len(self.dates)))
                         if (self.dates[i] - self.dates[i - 1]).total_seconds() > 0]
                data_time_step = int(np.median(diffs)) if diffs else 15
            else:
                data_time_step = 15

            self.components, T, points_per_day, dates = self.analyzer.decompose(
                self.values, self.dates, data_time_step
            )

            self.values = self.values[:T]
            self.dates = dates

            self.es_core.calculate_all_percentiles(
                original=self.values,
                comp_11yr=self.components['11yr'],
                comp_annual=self.components['annual'],
                comp_seasonal=self.components['seasonal'],
                comp_diurnal=self.components['diurnal']
            )

            self.root.after(0, lambda: messagebox.showinfo(
                self.config.get_label('success'),
                self.config.get_label('spectral_success')
            ))

        threading.Thread(target=analyze, daemon=True).start()
        messagebox.showinfo(self.config.get_label('info'), self.config.get_label('spectral_info'))

    def show_results(self):
        if len(self.dates) == 0:
            messagebox.showwarning(self.config.get_label('warning'), self.config.get_label('no_analysis_warning'))
            return

        self.clear_frame()

        top_frame = ttk.Frame(self.main_frame)
        top_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(top_frame, text=self.config.get_label('btn_back'),
                   command=self.show_main_page).pack(side=tk.LEFT, padx=10)
        ttk.Label(top_frame, text=self.config.get_label('title_results'),
                  font=("Arial", 14, "bold")).pack(side=tk.LEFT, expand=True)

        select_frame = ttk.LabelFrame(self.main_frame, text=self.config.get_label('title_select_point'), padding=10)
        select_frame.pack(fill=tk.X, padx=10, pady=5)

        datetime_frame = ttk.Frame(select_frame)
        datetime_frame.pack(pady=5)

        ttk.Label(datetime_frame, text=self.config.get_label('label_year')).pack(side=tk.LEFT, padx=2)
        self.year_var = tk.StringVar()
        years = sorted(list(set([d.year for d in self.dates])))
        year_combo = ttk.Combobox(datetime_frame, textvariable=self.year_var, values=years, width=6, state="readonly")
        year_combo.pack(side=tk.LEFT, padx=5)
        year_combo.bind('<<ComboboxSelected>>', lambda e: self.update_days_combo())

        ttk.Label(datetime_frame, text=self.config.get_label('label_month')).pack(side=tk.LEFT, padx=2)
        self.month_var = tk.StringVar()
        months = list(range(1, 13))
        month_combo = ttk.Combobox(datetime_frame, textvariable=self.month_var, values=months, width=4,
                                   state="readonly")
        month_combo.pack(side=tk.LEFT, padx=5)
        month_combo.bind('<<ComboboxSelected>>', lambda e: self.update_days_combo())

        ttk.Label(datetime_frame, text=self.config.get_label('label_day')).pack(side=tk.LEFT, padx=2)
        self.day_var = tk.StringVar()
        self.day_combo = ttk.Combobox(datetime_frame, textvariable=self.day_var, values=[], width=4, state="readonly")
        self.day_combo.pack(side=tk.LEFT, padx=5)
        self.day_combo.bind('<<ComboboxSelected>>', lambda e: self.update_time_combo())

        ttk.Label(datetime_frame, text=self.config.get_label('label_hour')).pack(side=tk.LEFT, padx=2)
        self.hour_var = tk.StringVar()
        self.hour_combo = ttk.Combobox(datetime_frame, textvariable=self.hour_var, values=[], width=4, state="readonly")
        self.hour_combo.pack(side=tk.LEFT, padx=5)
        self.hour_combo.bind('<<ComboboxSelected>>', lambda e: self.update_minute_combo())

        ttk.Label(datetime_frame, text=self.config.get_label('label_minute')).pack(side=tk.LEFT, padx=2)
        self.minute_var = tk.StringVar()
        self.minute_combo = ttk.Combobox(datetime_frame, textvariable=self.minute_var, values=[], width=4,
                                         state="readonly")
        self.minute_combo.pack(side=tk.LEFT, padx=5)

        def on_select():
            try:
                year = int(self.year_var.get())
                month = int(self.month_var.get())
                day = int(self.day_var.get())
                hour = int(self.hour_var.get())
                minute = int(self.minute_var.get())

                selected_date = datetime(year, month, day, hour, minute)

                closest_idx = -1
                min_diff = float('inf')
                for i, d in enumerate(self.dates):
                    diff = abs((d - selected_date).total_seconds())
                    if diff < min_diff:
                        min_diff = diff
                        closest_idx = i

                if closest_idx != -1 and min_diff < 3600:
                    self.selected_point_idx = closest_idx
                    self.update_analysis_tab()
                else:
                    messagebox.showwarning(self.config.get_label('warning'), self.config.get_label('no_point_warning'))
            except Exception as e:
                messagebox.showerror(self.config.get_label('error'),
                                     f"{self.config.get_label('invalid_date_error')}\n{str(e)}")

        ttk.Button(datetime_frame, text=self.config.get_label('btn_select_point'), command=on_select).pack(side=tk.LEFT,
                                                                                                           padx=20)

        nav_frame = ttk.Frame(select_frame)
        nav_frame.pack(pady=5)

        def prev_point():
            if self.selected_point_idx > 0:
                self.selected_point_idx -= 1
                self.update_analysis_tab()

        def next_point():
            if self.selected_point_idx < len(self.dates) - 1:
                self.selected_point_idx += 1
                self.update_analysis_tab()

        ttk.Button(nav_frame, text=self.config.get_label('btn_prev'), command=prev_point).pack(side=tk.LEFT, padx=5)

        self.point_info_var = tk.StringVar(value="")
        ttk.Label(nav_frame, textvariable=self.point_info_var).pack(side=tk.LEFT, padx=10)

        ttk.Button(nav_frame, text=self.config.get_label('btn_next'), command=next_point).pack(side=tk.LEFT, padx=5)

        self.results_notebook = ttk.Notebook(self.main_frame)
        self.results_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.analysis_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.analysis_frame, text="Анализ по правилам ЭС (Таблицы 1-6)")

        if self.components:
            components_frame = ttk.Frame(self.results_notebook)
            self.results_notebook.add(components_frame, text="Спектральные компоненты")
            self.plot_components(components_frame)

        self.selected_point_idx = len(self.dates) - 1
        last_date = self.dates[-1]
        self.year_var.set(str(last_date.year))
        self.month_var.set(str(last_date.month))
        self.update_days_combo()
        self.day_var.set(str(last_date.day))
        self.update_time_combo()
        self.hour_var.set(str(last_date.hour))
        self.update_minute_combo()
        self.minute_var.set(str(last_date.minute))

        self.update_analysis_tab()

    def update_days_combo(self):
        try:
            year = int(self.year_var.get())
            month = int(self.month_var.get())

            if month in [1, 3, 5, 7, 8, 10, 12]:
                days = 31
            elif month in [4, 6, 9, 11]:
                days = 30
            else:
                if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
                    days = 29
                else:
                    days = 28

            available_days = []
            for day in range(1, days + 1):
                for d in self.dates:
                    if d.year == year and d.month == month and d.day == day:
                        available_days.append(day)
                        break

            self.day_combo['values'] = available_days
            if available_days:
                if int(self.day_var.get()) not in available_days:
                    self.day_var.set(str(available_days[0]))
            else:
                self.day_var.set("")
        except:
            pass

    def update_time_combo(self):
        try:
            year = int(self.year_var.get())
            month = int(self.month_var.get())
            day = int(self.day_var.get())

            available_hours = []
            for hour in range(0, 24):
                for d in self.dates:
                    if d.year == year and d.month == month and d.day == day and d.hour == hour:
                        available_hours.append(hour)
                        break

            self.hour_combo['values'] = available_hours
            if available_hours:
                if self.hour_var.get() and int(self.hour_var.get()) in available_hours:
                    pass
                else:
                    self.hour_var.set(str(available_hours[0]))
            else:
                self.hour_var.set("")

            self.update_minute_combo()
        except:
            pass

    def update_minute_combo(self):
        try:
            year = int(self.year_var.get())
            month = int(self.month_var.get())
            day = int(self.day_var.get())
            hour = int(self.hour_var.get())

            available_minutes = []
            for minute in range(0, 60):
                for d in self.dates:
                    if d.year == year and d.month == month and d.day == day and d.hour == hour and d.minute == minute:
                        available_minutes.append(minute)
                        break

            self.minute_combo['values'] = available_minutes
            if available_minutes:
                if self.minute_var.get() and int(self.minute_var.get()) in available_minutes:
                    pass
                else:
                    self.minute_var.set(str(available_minutes[0]))
            else:
                self.minute_var.set("")
        except:
            pass

    def plot_components(self, parent):
        fig = Figure(figsize=(12, 10), dpi=100)

        components = [
            (self.components['11yr'], self.config.get_label('component_11yr')),
            (self.components['annual'], self.config.get_label('component_annual')),
            (self.components['seasonal'], self.config.get_label('component_seasonal')),
            (self.components['diurnal'], self.config.get_label('component_diurnal')),
            (self.components['hf'], self.config.get_label('component_hf'))
        ]

        for i, (comp, name) in enumerate(components):
            ax = fig.add_subplot(3, 2, i + 1)
            ax.plot(self.dates, comp, 'b-', linewidth=0.8)
            ax.set_title(name, fontsize=10)
            ax.set_ylabel('МГц', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45, labelsize=8)

        ax = fig.add_subplot(3, 2, 6)
        ax.plot(self.dates, self.values, 'b-', linewidth=0.8)
        ax.set_title(self.config.get_label('signal_original'), fontsize=10)
        ax.set_xlabel('Дата', fontsize=9)
        ax.set_ylabel('МГц', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45, labelsize=8)

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_analysis_tab(self):
        for widget in self.analysis_frame.winfo_children():
            widget.destroy()

        idx = self.selected_point_idx
        date = self.dates[idx]
        original_value = self.values[idx]

        if self.components:
            comp_11yr = self.components['11yr'][idx]
            comp_annual = self.components['annual'][idx]
            comp_seasonal = self.components['seasonal'][idx]
            comp_diurnal = self.components['diurnal'][idx]
        else:
            comp_11yr = original_value
            comp_annual = original_value
            comp_seasonal = original_value
            comp_diurnal = original_value

        prev_value = self.values[idx - 1] if idx > 0 else None
        next_value = self.values[idx + 1] if idx < len(self.values) - 1 else None

        prev_phase = None
        prev_phase_conf = 1.0
        if len(self.es_core.phase_history) > 0:
            prev_phase = self.es_core.phase_history[-1].get('phase')
            prev_phase_conf = self.es_core.phase_history[-1].get('conf', 1.0)

        solar_phase, solar_conf, solar_exp = self.es_core.analyze_solar_phase(
            comp_11yr, idx, prev_phase=prev_phase, prev_phase_conf=prev_phase_conf, component='11yr'
        )
        annual_state, annual_conf, annual_exp = self.es_core.analyze_annual_anomaly(comp_annual, date.month)
        seasonal_state, seasonal_conf, seasonal_exp = self.es_core.analyze_seasonal_cycle(comp_seasonal, date.month)
        diurnal_state, diurnal_conf, diurnal_exp = self.es_core.analyze_diurnal_cycle(comp_diurnal, date.hour)
        anomaly_type, anomaly_conf, anomaly_exp = self.es_core.detect_anomaly(original_value, prev_value, next_value)

        pred_extremum, pred_years, pred_conf, pred_exp = self.es_core.predict_extremum(solar_phase)

        self.es_core.phase_history.append({'phase': solar_phase, 'conf': solar_conf})
        if len(self.es_core.phase_history) > 20:
            self.es_core.phase_history.pop(0)

        point_info = self.config.get_label('point_info',
                                           idx=idx + 1,
                                           total=len(self.dates),
                                           date=date.strftime('%Y-%m-%d %H:%M:%S'))
        self.point_info_var.set(point_info)

        self.year_var.set(str(date.year))
        self.month_var.set(str(date.month))
        self.update_days_combo()
        self.day_var.set(str(date.day))
        self.hour_var.set(str(date.hour))
        self.minute_var.set(str(date.minute))

        canvas = tk.Canvas(self.analysis_frame)
        scrollbar = ttk.Scrollbar(self.analysis_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        ttk.Label(scrollable_frame, text=f"Точка: {date.strftime('%Y-%m-%d %H:%M:%S')}",
                  font=("Arial", 14, "bold")).pack(pady=10)
        ttk.Label(scrollable_frame, text=f"foF2 = {original_value:.2f} МГц", font=("Arial", 12)).pack()

        if self.components:
            ttk.Label(scrollable_frame,
                      text=f"11-летняя={comp_11yr:.2f} | Годовая={comp_annual:.2f} | Сезонная={comp_seasonal:.2f} | Суточная={comp_diurnal:.2f} МГц",
                      font=("Arial", 9), foreground="gray").pack(pady=5)

        frame1 = ttk.LabelFrame(scrollable_frame, text=f"Таблица 1. {self.config.get_table_title('table1')}",
                                padding=10)
        frame1.pack(fill=tk.X, pady=5, padx=10)
        ttk.Label(frame1, text=f"Фаза: {self.config.get_phase_label(solar_phase)}", font=("Arial", 11, "bold")).pack(
            anchor=tk.W)
        ttk.Label(frame1, text=f"Уверенность: {solar_conf:.0%}").pack(anchor=tk.W)
        ttk.Label(frame1, text=solar_exp, wraplength=700, foreground="gray").pack(anchor=tk.W, pady=5)

        frame2 = ttk.LabelFrame(scrollable_frame, text=f"Таблица 2. {self.config.get_table_title('table2')}",
                                padding=10)
        frame2.pack(fill=tk.X, pady=5, padx=10)
        ttk.Label(frame2, text=f"Состояние: {self.config.get_annual_label(annual_state)}",
                  font=("Arial", 11, "bold")).pack(anchor=tk.W)
        ttk.Label(frame2, text=f"Уверенность: {annual_conf:.0%}").pack(anchor=tk.W)
        ttk.Label(frame2, text=annual_exp, wraplength=700, foreground="gray").pack(anchor=tk.W, pady=5)

        frame3 = ttk.LabelFrame(scrollable_frame, text=f"Таблица 3. {self.config.get_table_title('table3')}",
                                padding=10)
        frame3.pack(fill=tk.X, pady=5, padx=10)
        ttk.Label(frame3, text=f"Фаза: {self.config.get_seasonal_label(seasonal_state)}",
                  font=("Arial", 11, "bold")).pack(anchor=tk.W)
        ttk.Label(frame3, text=f"Уверенность: {seasonal_conf:.0%}").pack(anchor=tk.W)
        ttk.Label(frame3, text=seasonal_exp, wraplength=700, foreground="gray").pack(anchor=tk.W, pady=5)

        frame4 = ttk.LabelFrame(scrollable_frame, text=f"Таблица 4. {self.config.get_table_title('table4')}",
                                padding=10)
        frame4.pack(fill=tk.X, pady=5, padx=10)
        ttk.Label(frame4, text=f"Фаза: {self.config.get_diurnal_label(diurnal_state)}",
                  font=("Arial", 11, "bold")).pack(anchor=tk.W)
        ttk.Label(frame4, text=f"Уверенность: {diurnal_conf:.0%}").pack(anchor=tk.W)
        ttk.Label(frame4, text=diurnal_exp, wraplength=700, foreground="gray").pack(anchor=tk.W, pady=5)

        frame5 = ttk.LabelFrame(scrollable_frame, text=f"Таблица 5. {self.config.get_table_title('table5')}",
                                padding=10)
        frame5.pack(fill=tk.X, pady=5, padx=10)
        if anomaly_type != 'NORMAL':
            ttk.Label(frame5, text=f"Тип: {self.config.get_anomaly_label(anomaly_type)}",
                      foreground="red", font=("Arial", 11, "bold")).pack(anchor=tk.W)
        else:
            ttk.Label(frame5, text=f"Тип: {self.config.get_anomaly_label(anomaly_type)}",
                      foreground="green", font=("Arial", 11, "bold")).pack(anchor=tk.W)
        ttk.Label(frame5, text=f"Уверенность: {anomaly_conf:.0%}").pack(anchor=tk.W)
        ttk.Label(frame5, text=anomaly_exp, wraplength=700, foreground="gray").pack(anchor=tk.W, pady=5)

        if pred_extremum:
            frame6 = ttk.LabelFrame(scrollable_frame, text=f"Таблица 6. {self.config.get_table_title('table6')}",
                                    padding=10)
            frame6.pack(fill=tk.X, pady=5, padx=10)
            extremum_text = self.config.get_extremum_label(pred_extremum)
            ttk.Label(frame6, text=f"Прогноз: {extremum_text} через {pred_years:.1f} лет",
                      font=("Arial", 11, "bold"), foreground="blue").pack(anchor=tk.W)
            ttk.Label(frame6, text=f"Уверенность: {pred_conf:.0%}").pack(anchor=tk.W)
            ttk.Label(frame6, text=pred_exp, wraplength=700, foreground="gray").pack(anchor=tk.W, pady=5)

        stats_frame = ttk.LabelFrame(scrollable_frame, text=self.config.get_label('stats_title'), padding=10)
        stats_frame.pack(fill=tk.X, pady=5, padx=10)

        for comp_name, ru_name in [('11yr', '11-летняя'), ('annual', 'Годовая'), ('seasonal', 'Сезонная'),
                                   ('diurnal', 'Суточная')]:
            p10 = self.es_core.percentiles[comp_name]['p10']
            p25 = self.es_core.percentiles[comp_name]['p25']
            p50 = self.es_core.percentiles[comp_name]['p50']
            p75 = self.es_core.percentiles[comp_name]['p75']
            p90 = self.es_core.percentiles[comp_name]['p90']
            if p10 is not None:
                ttk.Label(stats_frame,
                          text=f"{ru_name}: p10={p10:.2f} | p25={p25:.2f} | p50={p50:.2f} | p75={p75:.2f} | p90={p90:.2f} МГц",
                          font=("Arial", 9)).pack(anchor=tk.W)

        ttk.Button(scrollable_frame, text=self.config.get_label('btn_main_menu'),
                   command=self.show_main_page).pack(pady=20)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def show_knowledge_base(self):
        self.clear_frame()

        canvas = tk.Canvas(self.main_frame)
        scrollbar = ttk.Scrollbar(self.main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        ttk.Label(scrollable_frame, text=self.config.get_label('title_knowledge'),
                  font=("Arial", 16, "bold")).pack(pady=10)

        tables = ['table1', 'table2', 'table3', 'table4', 'table5', 'table6']

        for i, table_key in enumerate(tables, 1):
            rules = self.config.get_table_rules(table_key)
            title = self.config.get_table_title(table_key)

            if not rules:
                continue

            frame = ttk.LabelFrame(scrollable_frame, text=f"Таблица {i}. {title}", padding=10)
            frame.pack(fill=tk.X, pady=5, padx=20)

            for rule in rules:
                condition = rule.get('condition', '')
                result_key = rule.get('result', '')
                confidence = rule.get('confidence', 0)
                years = rule.get('years', None)

                if table_key == 'table6':
                    result_label = self.config.get_extremum_label(result_key)
                    if years:
                        result_text = f"{result_label} через {years} лет"
                    else:
                        result_text = result_label
                elif table_key == 'table1':
                    result_text = self.config.get_phase_label(result_key)
                elif table_key == 'table2':
                    result_text = self.config.get_annual_label(result_key)
                elif table_key == 'table3':
                    result_text = self.config.get_seasonal_label(result_key)
                elif table_key == 'table4':
                    result_text = self.config.get_diurnal_label(result_key)
                elif table_key == 'table5':
                    result_text = self.config.get_anomaly_label(result_key)
                else:
                    result_text = result_key

                ttk.Label(frame, text=f"• ЕСЛИ {condition} → ТО {result_text} (уверенность: {confidence})",
                          wraplength=800).pack(anchor=tk.W, pady=2)

        ttk.Button(scrollable_frame, text=self.config.get_label('btn_back'),
                   command=self.show_main_page).pack(pady=20)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = IonosphereExpertSystem()
    app.run()