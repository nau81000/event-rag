import { bootstrapApplication } from '@angular/platform-browser';
import { Component, ElementRef, ViewChild, NgZone, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

import { provideHttpClient } from '@angular/common/http';
import { provideTranslateService } from '@ngx-translate/core';
import { provideTranslateHttpLoader } from '@ngx-translate/http-loader';

import { TranslateService, TranslatePipe } from '@ngx-translate/core';

import { Subscription } from 'rxjs';
import { ChatSocketService } from './app/websocket.service';
import { GeolocationService } from './app/geolocation.service';
import { ChatHistoryService } from './app/chat-history.service';

interface ChatMessage {
  role: 'user' | 'assistant';
  text: string;
}

interface SearchResult {
  payload: any; // or a proper type if you have one
}

interface ChatResponse {
  answer: SearchResult[];
}

export interface Timing {
  begin: string; // ISO date string
  end: string;   // ISO date string
}

export interface EventPayload {
  title_fr?: string;
  location_city?: string;
  location_department?: string;
  location_countrycode?: string;
  conditions_fr?: string;
  timings?: string | null; // JSON string of Timing[]
}

/**
 * Group contiguous days that share the same time interval.
 * Input: JSON string of Timing[]
 * Output: human-readable French strings
 */
export function groupTimings(timingsStr: string | null | undefined): string[] {
  if (!timingsStr) {
    return ['Dates non précisées'];
  }

  let timings: Timing[];
  try {
    timings = JSON.parse(timingsStr) as Timing[];
  } catch {
    return ['Dates non lisibles'];
  }

  if (!Array.isArray(timings) || timings.length === 0) {
    return ['Dates non précisées'];
  }

  // Parse ISO timestamps into Date objects
  let slots = timings.map((t) => {
    const begin = new Date(t.begin);
    const end = new Date(t.end);
    return { begin, end };
  });

  if (!slots.length) {
    return ['Dates non précisées'];
  }

  // Sort by begin date
  slots.sort((a, b) => a.begin.getTime() - b.begin.getTime());

  const groups: {
    start: Date;
    end: Date;
    hb: string; // hour begin
    he: string; // hour end
  }[] = [];

  // Format hour as "HHhMM"
  const fmtHour = (d: Date) =>
    d
      .toLocaleTimeString('fr-FR', {
        hour: '2-digit',
        minute: '2-digit',
        hour12: false,
      })
      .replace(':', 'h');

  let currentStart = slots[0].begin;
  let currentEnd = slots[0].end;
  let currentBeginTime = fmtHour(currentStart);
  let currentEndTime = fmtHour(currentEnd);

  for (let i = 1; i < slots.length; i++) {
    const begin = slots[i].begin;
    const end = slots[i].end;

    const isNextDay =
      begin.getDate() === currentEnd.getDate() + 1 &&
      begin.getMonth() === currentEnd.getMonth() &&
      begin.getFullYear() === currentEnd.getFullYear();

    const sameTime =
      fmtHour(begin) === currentBeginTime && fmtHour(end) === currentEndTime;

    if (isNextDay && sameTime) {
      // Extend group
      currentEnd = end;
    } else {
      // Close current group
      groups.push({
        start: currentStart,
        end: currentEnd,
        hb: currentBeginTime,
        he: currentEndTime,
      });

      currentStart = begin;
      currentEnd = end;
      currentBeginTime = fmtHour(begin);
      currentEndTime = fmtHour(end);
    }
  }

  // Push last group
  groups.push({
    start: currentStart,
    end: currentEnd,
    hb: currentBeginTime,
    he: currentEndTime,
  });

  // Format final human-readable output
  const humanGroups: string[] = [];

  const fmtDate = (d: Date) =>
    d.toLocaleDateString('fr-FR', {
      weekday: 'long',
      day: '2-digit',
      month: 'long',
      year: 'numeric',
    });

  groups.forEach((g) => {
    const startStr = fmtDate(g.start);
    const endStr = fmtDate(g.end);

    if (startStr === endStr) {
      humanGroups.push(`Le ${startStr} — ${g.hb} à ${g.he}`);
    } else {
      humanGroups.push(`Du ${startStr} au ${endStr} — ${g.hb} à ${g.he}`);
    }
  });

  return humanGroups;
}

/**
 * Format a single event payload into a multi-line string for the chatbot.
 */
export function formatEvent(payload: EventPayload): string {
  const p = payload || {};

  const title = p.title_fr ?? 'Titre non précisé';
  const city = p.location_city || 'Non précisé';
  const dep = p.location_department ?? '?';
  const country = p.location_countrycode ?? '?';
  const conditions = p.conditions_fr;

  const dates = (groupTimings(p.timings) || [])
    .map((line) => `    ${line}`)
    .join('\n');

  return `
- ${title}
    📍 ${city} (${dep}, ${country})
    📅 Dates :
${dates}
    🎟 ${conditions ? conditions : "Conditions d'accès non précisées"}
  `.trim();
}

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [FormsModule, CommonModule, TranslatePipe],
  template: `
    <div class="layout">
      
      <!-- SIDEBAR -->
      <div class="sidebar">
        <img src="events.png" alt="Event image">
        <div class="sidebar-footer">
          {{ 'sidebar.footer' | translate }}
        </div>
      </div>

      <main class="content">
        <div class="header">
          <h5 class="header-title">{{ 'main.title' | translate}}</h5>
          <div class="lang-menu-wrapper">
            <svg
              class="globe-icon"
              (click)="toggleLangMenu()"
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="2"
              stroke-linecap="round"
              stroke-linejoin="round"
            >
              <circle cx="12" cy="12" r="10"></circle>
              <line x1="2" y1="12" x2="22" y2="12"></line>
              <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"></path>
            </svg>
            @if (langMenuOpen) {
              <div class="lang-menu">
                <button
                  type="button"
                  class="lang-item"
                  [class.active]="currentLang === 'en'"
                  (click)="setLang('en')"
                >
                  English
                </button>
                <button
                  type="button"
                  class="lang-item"
                  [class.active]="currentLang === 'fr'"
                  (click)="setLang('fr')"
                >
                  Français
                </button>
              </div>
            }        
          </div>
        </div>

        <!-- CHAT AREA -->
        <div class="chat-container">
          <div class="chat-messages" #messagesContainer>
            <div
              *ngFor="let msg of messages"
              class="chat-message"
              [class.user]="msg.role === 'user'"
              [class.assistant]="msg.role === 'assistant'"
            >
              <div class="chat-bubble">
                {{ msg.text }}
              </div>
            </div>
          </div>
          <form class="chat-input-row" (ngSubmit)="sendMessage()">
            @if (city) {
              <button
                class="spark-btn"
                type="button"
                (click)="searchOnGeolocation()"
                title="Chercher proche de moi"
              >
                ❇️
              </button>
            }
            <input
              type="text"
              name="message"
              [(ngModel)]="currentInput"
              class="chat-input"
              placeholder="{{ 'main.input_placeholder' | translate}}"
              autocomplete="off"
              (keydown)="handleKeyDown($event)"
            />
          </form>
        </div>
      <!-- MAIN AREA -->
      </main>
    </div>
  `,
})

export class App {
  @ViewChild('messagesContainer') messagesContainer?: ElementRef<HTMLDivElement>;
  messages: ChatMessage[] = [];
  currentInput = '';
  langMenuOpen = false;
  currentLang: 'en' | 'fr' = 'fr';
  city: string | null = null;
  history: string[] = [];     // loaded from IndexedDB
  historyIndex = -1;          // -1 means "not in history"
  private socketSub?: Subscription;
  private readonly LANG_KEY = 'app-lang';

  constructor(
    private translate: TranslateService,
    private chatSocket: ChatSocketService,
    private ngZone: NgZone,
    private cdr: ChangeDetectorRef,
    private geo: GeolocationService,
    private chatHistory: ChatHistoryService
  ) {
    const initialLang = this.getInitialLang();
    this.currentLang = initialLang;
  }

  async detectCityOnInit() {
    try {
      const city = await this.geo.getCityFromBrowser();
      this.city = city;
      this.cdr.detectChanges();   // FORCE Angular to update now
    } catch (err: any) {
      console.warn("Unable to detect city on init:", err.message);
    }
  }

  navigateHistory(direction: -1 | 1) {
    if (this.history.length === 0) return;

    if (this.historyIndex === -1) {
      // Starting history navigation
      if (direction === -1) {
        // UP: jump to last request
        this.historyIndex = this.history.length - 1;
        this.currentInput = this.history[this.historyIndex];
      }
      return;
    }

    // Move index
    if (direction === 1 || this.historyIndex !== 0 ) {
      this.historyIndex += direction;
    }

    if (this.historyIndex >= this.history.length) {
      this.historyIndex = this.history.length - 1;
      this.currentInput = '';
    } else {
      // Set input to history entry
      this.currentInput = this.history[this.historyIndex];
    }

  }

  handleKeyDown(event: KeyboardEvent) {
    if (event.key === 'ArrowUp') {
      event.preventDefault();
      this.navigateHistory(-1); // go back
    } else if (event.key === 'ArrowDown') {
      event.preventDefault();
      this.navigateHistory(1); // go forward
    } else if (event.key === 'Enter') {
      this.sendMessage();
    }
  }

  toggleLangMenu() {
    this.langMenuOpen = !this.langMenuOpen;
  }

  setLang(lang: 'en' | 'fr') {
    this.currentLang = lang;
  
    // Save in local storage
    localStorage.setItem(this.LANG_KEY, lang);

    // Update first assistant greeting if present
    this.translate.use(lang).subscribe(() => {
      this.translate.get('main.welcome_message').subscribe((greeting) => {
        if (this.messages.length && this.messages[0].role === 'assistant') {
          this.messages[0].text = greeting;
        }
      });
    });
    this.langMenuOpen = false;
    this.cdr.detectChanges();   // FORCE Angular to update now
  }
  
  async ngOnInit() {
    // Detect city
    this.detectCityOnInit();
    // Websocket connection
    this.chatSocket.connect();

    this.translate.use(this.currentLang).subscribe(async () => {
      // Load last user requests from IndexedDB
      const previousRequests = await this.chatHistory.loadLastRequests(50);

      // Save into memory for arrow navigation
      this.history = previousRequests;
      this.historyIndex = -1;

      this.translate.get('main.welcome_message').subscribe(text => {
        this.messages.push({ role: 'assistant', text });
      });
    });

    // Subscribe to incoming messages (from server)
    this.socketSub = this.chatSocket.messages$.subscribe((raw) => {
      if (!raw) return;

      this.ngZone.run(() => {
        const data = JSON.parse(raw) as ChatResponse;

        // Map SearchResult[] -> formatted strings
        const items = data.answer.map((res) => formatEvent(res.payload));

        const text =
          items.length > 0
            ? items.join('\n\n')
            : this.translate.instant('searchResult'); // equivalent to t("searchResult")

        const assistantMsg: ChatMessage = {
          role: 'assistant',
          text,
        };

        this.messages.push(assistantMsg);
        this.scrollToBottom();
      });
    });
  }

  ngOnDestroy() {
    this.socketSub?.unsubscribe();
    this.chatSocket.close();
  }

  private scrollToBottom() {
    if (!this.messagesContainer) return;

    const el = this.messagesContainer?.nativeElement;
    if (!el) return;
    this.cdr.detectChanges();
    el.scrollTop = el.scrollHeight;
  }

  private getInitialLang(): 'fr' | 'en' {
    // Check localStorage
    const stored = localStorage.getItem(this.LANG_KEY);
    if (stored === 'fr' || stored === 'en') {
      return stored;
    }

    const lang = navigator.language.toLowerCase(); // e.g. 'fr-FR', 'en-US'

    if (lang.startsWith('fr')) return 'fr';
    return 'en';
  }

  private persistUserRequests(text: string) {
    this.history.push(text);
    this.chatHistory.saveRequests(this.history)
      .catch((err) =>
        console.error('Failed to save user requests', err)
      );
    this.historyIndex = -1;
  }

  sendMessage() {
    const text = this.currentInput.trim();
    if (!text) return;

    // add user message
    this.messages.push({ role: 'user', text });

    // Send to WebSocket server
    this.chatSocket.send(text);

    this.currentInput = '';

    // Save request
    this.persistUserRequests(text);
  }

  searchOnGeolocation() {
    if (!this.city) return;

    const request = this.city + "?"; 

    // add user message
    this.messages.push({ role: 'user', text: request });

    // Save request
    this.persistUserRequests(request);

    // Send to WebSocket server
    this.chatSocket.send(this.city);
  }  
}

bootstrapApplication(App,
  {
    providers: [
      provideHttpClient(),
      provideTranslateService({
        loader: provideTranslateHttpLoader({
          prefix: 'i18n/',   // served from public/i18n/
          suffix: '.json'
        })
      })
    ]
  }
);