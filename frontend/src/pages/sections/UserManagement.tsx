/**
 * AuthFaceGraph — User Management Section
 * Identity profiles, roles, and session history
 */

import React, { useState } from 'react';
import { Users, Search, Shield, Clock, Eye, Edit, UserPlus } from 'lucide-react';
import { GlassCard, SectionHeader, NeonButton, StatusDot } from '../../components/ui';
import { useAuthStore } from '../../store';

interface UserRecord {
  id: string;
  name: string;
  email: string;
  role: string;
  status: 'online' | 'offline';
  lastSeen: string;
  sessions: number;
}

const DEMO_USERS: UserRecord[] = [
  { id:'u001', name:'Dr. Mehedi Hassan',  email:'mehedi@authfacegraph.ai', role:'admin',    status:'online',  lastSeen:'Now',       sessions:47 },
  { id:'u002', name:'Sarah Chen',          email:'sarah@authfacegraph.ai',  role:'analyst',  status:'offline', lastSeen:'2h ago',    sessions:23 },
  { id:'u003', name:'James Okafor',        email:'james@authfacegraph.ai',  role:'viewer',   status:'offline', lastSeen:'1d ago',    sessions:8  },
  { id:'u004', name:'Priya Sharma',        email:'priya@authfacegraph.ai',  role:'analyst',  status:'online',  lastSeen:'5m ago',    sessions:31 },
  { id:'u005', name:'Lucas Müller',        email:'lucas@authfacegraph.ai',  role:'viewer',   status:'offline', lastSeen:'3d ago',    sessions:4  },
];

const ROLE_COLORS: Record<string,string> = {
  admin:   '#ef4444',
  analyst: '#00d4ff',
  viewer:  '#8b5cf6',
};

export const UserManagement: React.FC = () => {
  const [search, setSearch] = useState('');
  const { role: myRole } = useAuthStore();

  const filtered = DEMO_USERS.filter(u =>
    u.name.toLowerCase().includes(search.toLowerCase()) ||
    u.email.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="space-y-5 stagger-children">
      <SectionHeader
        title="User Management"
        subtitle="Identity profiles, roles and access control"
        icon={<Users size={16} />}
        actions={
          myRole === 'admin' && (
            <NeonButton size="sm" variant="primary">
              <UserPlus size={13} /> Invite User
            </NeonButton>
          )
        }
      />

      {/* Stats Row */}
      <div className="grid grid-cols-3 gap-4">
        {[
          { label: 'Total Users',    value: DEMO_USERS.length,                               color: '#00d4ff' },
          { label: 'Online Now',     value: DEMO_USERS.filter(u=>u.status==='online').length, color: '#10b981' },
          { label: 'Total Sessions', value: DEMO_USERS.reduce((a,u)=>a+u.sessions,0),        color: '#8b5cf6' },
        ].map((s, i) => (
          <GlassCard key={i} className="p-4 text-center">
            <div className="font-bold text-2xl" style={{ color: s.color }}>{s.value}</div>
            <div className="font-mono text-[9px] text-slate-400 uppercase tracking-widest mt-1">{s.label}</div>
          </GlassCard>
        ))}
      </div>

      {/* Search */}
      <GlassCard className="p-4">
        <div className="relative">
          <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
          <input
            type="text"
            placeholder="Search users by name or email..."
            value={search}
            onChange={e => setSearch(e.target.value)}
            className="neon-input w-full pl-9 pr-4 py-2.5 rounded-xl text-sm"
          />
        </div>
      </GlassCard>

      {/* User Table */}
      <GlassCard>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-indigo-500/10">
                {['User', 'Role', 'Status', 'Sessions', 'Last Active', 'Actions'].map(h => (
                  <th key={h} className="px-4 py-3 text-left font-mono text-[9px] uppercase tracking-widest text-slate-500">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {filtered.map((user, i) => (
                <tr
                  key={user.id}
                  className="border-b border-indigo-500/05 transition-colors"
                  style={{ animationDelay: `${i * 40}ms` }}
                  onMouseEnter={e => (e.currentTarget as HTMLElement).style.background = 'rgba(79,70,229,0.05)'}
                  onMouseLeave={e => (e.currentTarget as HTMLElement).style.background = 'transparent'}
                >
                  {/* User */}
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-3">
                      <div className="w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold"
                        style={{ background: 'linear-gradient(135deg,#4f46e5,#7c3aed)' }}>
                        {user.name.charAt(0)}
                      </div>
                      <div>
                        <div className="text-xs font-semibold text-white">{user.name}</div>
                        <div className="font-mono text-[9px] text-slate-400">{user.email}</div>
                      </div>
                    </div>
                  </td>
                  {/* Role */}
                  <td className="px-4 py-3">
                    <span className="font-mono text-[10px] px-2 py-0.5 rounded-full uppercase font-bold"
                      style={{
                        color: ROLE_COLORS[user.role],
                        background: `${ROLE_COLORS[user.role]}18`,
                        border: `1px solid ${ROLE_COLORS[user.role]}35`,
                      }}>
                      {user.role}
                    </span>
                  </td>
                  {/* Status */}
                  <td className="px-4 py-3">
                    <StatusDot status={user.status} label={user.status === 'online' ? 'Online' : 'Offline'} />
                  </td>
                  {/* Sessions */}
                  <td className="px-4 py-3">
                    <span className="font-mono text-xs text-cyan-400">{user.sessions}</span>
                  </td>
                  {/* Last active */}
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-1.5 font-mono text-[10px] text-slate-400">
                      <Clock size={10} />
                      {user.lastSeen}
                    </div>
                  </td>
                  {/* Actions */}
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-1">
                      <button className="p-1.5 rounded-lg hover:bg-cyan-500/10 transition-colors text-slate-400 hover:text-cyan-400">
                        <Eye size={12} />
                      </button>
                      {myRole === 'admin' && (
                        <button className="p-1.5 rounded-lg hover:bg-violet-500/10 transition-colors text-slate-400 hover:text-violet-400">
                          <Edit size={12} />
                        </button>
                      )}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </GlassCard>
    </div>
  );
};
