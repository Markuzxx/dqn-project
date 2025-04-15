
# dqn-project

Implementazione di un agente Deep Q-Learning per il controllo decisionale in un ambiente simulato. L'agente utilizza due reti neurali (policy e target) per approssimare la funzione Q e apprendere politiche ottimali basate su segnali di ricompensa. L'esplorazione è gestita per bilanciare la raccolta di nuove esperienze e lo sfruttamento delle conoscenze già acquisite. Il progetto fornisce una base solida, pronta per future estensioni con tecniche avanzate.

**Nota:** *questa è una bozza iniziale della relazione relativa al mio capolavoro. Il progetto è attivamente in sviluppo e disponibile al seguente link GitHub: `https://github.com/Markuzxx/dqn-project` La versione finale sarà caricata entro i tempi previsti.*

## Motivazione

Ho scelto di sviluppare un agente basato sul Deep Q-Learning per approfondire concretamente i meccanismi fondamentali dell’intelligenza artificiale, in particolare l’apprendimento per rinforzo. Questo progetto mi permette di combinare le mie conoscenze nella programmazione e nella matematica con l'interesse per i sistemi adattivi e l'ottimizzazione del comportamento tramite esperienze.

## Struttura del progetto

- `config/`: contiene i file di configurazione (es. `hyperparameters.yml`)
- `sessions/`: sessioni salvate
- `agent.py`: implementazione dell'agente DQN
- `environment.py`: ambiente simulato (eventualmente personalizzato)
- `evaluate.py`: avvia la valutazione dell'agente
- `models.py`: definizione della rete neurale
- `replay_buffer.py`: memoria per l’esperienza dell’agente
- `train.py`: avvia l'addestramento dell'agente
- `utils.py`: costanti, classi e funzioni di supporto

## Stato attuale

- [x] Ambiente virtuale configurato  
- [x] Struttura modulare pronta  
- [x] Rete neurale base e replay buffer funzionanti  
- [x] Algoritmo DQN in fase di test  
- [ ] Integrazione valutazione e salvataggio modelli  
- [ ] Ottimizzazione e logging

## Prospettive future

Il progetto è pensato per essere esteso con:

- Double DQN  
- Dueling DQN  
- Prioritized Experience Replay (PER)  
- Ambienti personalizzati più complessi
